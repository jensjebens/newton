# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FEM Thin Shell Solver — Phase A: Membrane energy with implicit Euler + PCG.

GPU-accelerated FEM thin shell solver for stiff thin materials (cardboard,
sheet metal, plastic). Uses St. Venant-Kirchhoff membrane energy on triangle
elements with implicit Euler time integration solved via Conjugate Gradient.

Architecture:
    - Membrane energy: StVK on triangle elements (deformation gradient F → Green strain E → forces)
    - Time integration: Implicit Euler (backward Euler)
    - Linear solve: Conjugate Gradient via warp.optim.linear.cg
    - Stiffness matrix: Assembled as BsrMatrix (3x3 blocks) via warp.sparse
"""

import numpy as np
import warp as wp
import warp.sparse as wps
from warp.optim.linear import cg, preconditioner

from newton._src.solvers.solver import SolverBase


# ---------------------------------------------------------------------------
# Warp kernels for membrane FEM
# ---------------------------------------------------------------------------


@wp.func
def _compute_deformation_gradient(
    p0: wp.vec3,
    p1: wp.vec3,
    p2: wp.vec3,
    Dm_inv: wp.mat22,
):
    """Compute deformation gradient F = Ds @ Dm_inv.

    Returns two column vectors (f_col0, f_col1) representing the 3x2 matrix F.
    """
    e1 = p1 - p0
    e2 = p2 - p0

    f_col0 = e1 * Dm_inv[0, 0] + e2 * Dm_inv[1, 0]
    f_col1 = e1 * Dm_inv[0, 1] + e2 * Dm_inv[1, 1]

    return f_col0, f_col1


@wp.func
def _stvk_energy_density(f0: wp.vec3, f1: wp.vec3):
    """Compute StVK strain from deformation gradient columns.

    Returns Green-Lagrange strain components (e00, e01, e11).
    """
    c00 = wp.dot(f0, f0)
    c01 = wp.dot(f0, f1)
    c11 = wp.dot(f1, f1)

    e00 = 0.5 * (c00 - 1.0)
    e01 = 0.5 * c01
    e11 = 0.5 * (c11 - 1.0)

    return e00, e01, e11


@wp.kernel
def _compute_membrane_forces(
    particle_q: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=wp.int32),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_areas: wp.array(dtype=float),
    mu: float,
    lmbda: float,
    # outputs
    forces: wp.array(dtype=wp.vec3),
):
    """Compute membrane elastic forces for all triangles.

    Uses StVK constitutive model:
        W = ∫ (μ‖E‖² + λ/2 (tr E)²) dA

    Force on vertex i:  f_i = -∂W/∂x_i

    We compute forces via finite difference of energy for robustness
    in this first implementation. Will switch to analytic gradients
    for performance later.
    """
    tid = wp.tid()

    v0 = tri_indices[tid, 0]
    v1 = tri_indices[tid, 1]
    v2 = tri_indices[tid, 2]

    p0 = particle_q[v0]
    p1 = particle_q[v1]
    p2 = particle_q[v2]

    Dm_inv = tri_poses[tid]
    area = tri_areas[tid]

    # Current edge vectors
    e1 = p1 - p0
    e2 = p2 - p0

    # Deformation gradient columns: F = Ds @ Dm_inv
    f0 = e1 * Dm_inv[0, 0] + e2 * Dm_inv[1, 0]
    f1 = e1 * Dm_inv[0, 1] + e2 * Dm_inv[1, 1]

    # Green strain E = ½(FᵀF - I)
    c00 = wp.dot(f0, f0)
    c01 = wp.dot(f0, f1)
    c11 = wp.dot(f1, f1)

    e00 = 0.5 * (c00 - 1.0)
    e01 = 0.5 * c01
    e11 = 0.5 * (c11 - 1.0)

    # Second Piola-Kirchhoff stress: S = 2μE + λ tr(E) I
    tr_E = e00 + e11
    s00 = 2.0 * mu * e00 + lmbda * tr_E
    s01 = 2.0 * mu * e01
    s11 = 2.0 * mu * e11 + lmbda * tr_E

    # Force = -area * F @ S @ Dm_inv^T (chain rule)
    # P = F @ S (First Piola-Kirchhoff stress, 3x2)
    p00 = f0 * s00 + f1 * s01
    p01 = f0 * s01 + f1 * s11

    # H = -area * P @ Dm_inv^T (3x2 @ 2x2 → 3x2, but we need per-vertex forces)
    # Force on v1 = -area * H[:, 0], force on v2 = -area * H[:, 1]
    # Force on v0 = -(force on v1 + force on v2)
    h0 = p00 * Dm_inv[0, 0] + p01 * Dm_inv[0, 1]
    h1 = p00 * Dm_inv[1, 0] + p01 * Dm_inv[1, 1]

    force1 = -area * h0
    force2 = -area * h1
    force0 = -(force1 + force2)

    wp.atomic_add(forces, v0, force0)
    wp.atomic_add(forces, v1, force1)
    wp.atomic_add(forces, v2, force2)


@wp.kernel
def _compute_membrane_stiffness_triplets(
    particle_q: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=wp.int32),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_areas: wp.array(dtype=float),
    mu: float,
    lmbda: float,
    eps: float,
    # outputs — triplets for BSR assembly
    triplet_rows: wp.array(dtype=wp.int32),
    triplet_cols: wp.array(dtype=wp.int32),
    triplet_vals: wp.array(dtype=wp.mat33),
):
    """Compute stiffness matrix entries via finite difference of forces.

    For each triangle, compute dF/dx via central differences for each of the
    9 DOFs (3 vertices × 3 components). This gives 9×9 block which we scatter
    into 3×3 blocks at the right (row, col) positions.

    Each triangle contributes 9 entries (3×3 vertex pairs).
    """
    tid = wp.tid()

    v0 = tri_indices[tid, 0]
    v1 = tri_indices[tid, 1]
    v2 = tri_indices[tid, 2]

    p0 = particle_q[v0]
    p1 = particle_q[v1]
    p2 = particle_q[v2]

    Dm_inv = tri_poses[tid]
    area = tri_areas[tid]

    # We'll compute dforce/dx numerically for each vertex/component
    # This gives us the tangent stiffness K = -dF/dx
    verts = wp.vec3(float(v0), float(v1), float(v2))

    # For each pair (i, j) of the 3 vertices, compute the 3×3 block K_ij
    for vi in range(3):
        for vj in range(3):
            K_block = wp.mat33(0.0)

            for comp in range(3):
                # Perturb vertex vj, component comp by +eps and -eps
                pp0 = p0
                pp1 = p1
                pp2 = p2
                pm0 = p0
                pm1 = p1
                pm2 = p2

                if vj == 0:
                    pp0 = _perturb(p0, comp, eps)
                    pm0 = _perturb(p0, comp, -eps)
                elif vj == 1:
                    pp1 = _perturb(p1, comp, eps)
                    pm1 = _perturb(p1, comp, -eps)
                else:
                    pp2 = _perturb(p2, comp, eps)
                    pm2 = _perturb(p2, comp, -eps)

                # Compute force on vertex vi for both perturbations
                fp = _triangle_force_on_vertex(vi, pp0, pp1, pp2, Dm_inv, area, mu, lmbda)
                fm = _triangle_force_on_vertex(vi, pm0, pm1, pm2, Dm_inv, area, mu, lmbda)

                # K[vi][vj][:, comp] = -(fp - fm) / (2*eps)
                df = (fp - fm) * (0.5 / eps)

                # Stiffness = -df/dx, so K = -df
                for row in range(3):
                    K_block[row, comp] = -df[row]

            # Output triplet index: 9 entries per triangle
            out_idx = tid * 9 + vi * 3 + vj
            if vi == 0:
                triplet_rows[out_idx] = v0
            elif vi == 1:
                triplet_rows[out_idx] = v1
            else:
                triplet_rows[out_idx] = v2

            if vj == 0:
                triplet_cols[out_idx] = v0
            elif vj == 1:
                triplet_cols[out_idx] = v1
            else:
                triplet_cols[out_idx] = v2

            triplet_vals[out_idx] = K_block


@wp.func
def _perturb(p: wp.vec3, comp: int, delta: float) -> wp.vec3:
    """Perturb a single component of a vec3."""
    result = p
    if comp == 0:
        result = wp.vec3(p[0] + delta, p[1], p[2])
    elif comp == 1:
        result = wp.vec3(p[0], p[1] + delta, p[2])
    else:
        result = wp.vec3(p[0], p[1], p[2] + delta)
    return result


@wp.func
def _triangle_force_on_vertex(
    vi: int,
    p0: wp.vec3,
    p1: wp.vec3,
    p2: wp.vec3,
    Dm_inv: wp.mat22,
    area: float,
    mu: float,
    lmbda: float,
) -> wp.vec3:
    """Compute the membrane force on vertex vi of a single triangle."""
    e1 = p1 - p0
    e2 = p2 - p0

    f0 = e1 * Dm_inv[0, 0] + e2 * Dm_inv[1, 0]
    f1 = e1 * Dm_inv[0, 1] + e2 * Dm_inv[1, 1]

    c00 = wp.dot(f0, f0)
    c01 = wp.dot(f0, f1)
    c11 = wp.dot(f1, f1)

    e00 = 0.5 * (c00 - 1.0)
    e01 = 0.5 * c01
    e11 = 0.5 * (c11 - 1.0)

    tr_E = e00 + e11
    s00 = 2.0 * mu * e00 + lmbda * tr_E
    s01 = 2.0 * mu * e01
    s11 = 2.0 * mu * e11 + lmbda * tr_E

    p00 = f0 * s00 + f1 * s01
    p01 = f0 * s01 + f1 * s11

    h0 = p00 * Dm_inv[0, 0] + p01 * Dm_inv[0, 1]
    h1 = p00 * Dm_inv[1, 0] + p01 * Dm_inv[1, 1]

    force1 = -area * h0
    force2 = -area * h1
    force0 = -(force1 + force2)

    if vi == 0:
        return force0
    elif vi == 1:
        return force1
    else:
        return force2


@wp.kernel
def _compute_gravity_forces(
    particle_f: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    gravity: wp.vec3,
):
    """Add gravity to force accumulator."""
    i = wp.tid()
    if particle_inv_mass[i] > 0.0:
        mass = 1.0 / particle_inv_mass[i]
        wp.atomic_add(particle_f, i, gravity * mass)


@wp.kernel
def _implicit_euler_rhs(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    elastic_forces: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
    # output
    rhs: wp.array(dtype=wp.vec3),
):
    """Compute RHS of implicit Euler system: b = M*v_n + dt*f(x_n).

    The implicit Euler equation is:
        (M + dt²K) Δv = dt * f(x_n + dt*v_n) + M*(v_n - v_current)

    For simplicity in Phase A, we use a semi-implicit approach:
        (M + dt²K) dv = dt * (f_elastic + f_gravity)
        v_{n+1} = v_n + dv
        x_{n+1} = x_n + dt * v_{n+1}
    """
    i = wp.tid()
    if particle_inv_mass[i] > 0.0:
        mass = 1.0 / particle_inv_mass[i]
        f_total = elastic_forces[i] + gravity * mass
        rhs[i] = dt * f_total
    else:
        rhs[i] = wp.vec3(0.0)


@wp.kernel
def _apply_mass_diagonal(
    particle_inv_mass: wp.array(dtype=float),
    dt: float,
    # in/out
    diag: wp.array(dtype=wp.mat33),
):
    """Add M/dt² to the diagonal blocks of the stiffness matrix.

    The system matrix is: A = M + dt²K
    In block form: A_ii = m_i * I + dt² * K_ii
    """
    i = wp.tid()
    if particle_inv_mass[i] > 0.0:
        mass = 1.0 / particle_inv_mass[i]
        mass_block = wp.mat33(
            mass, 0.0, 0.0,
            0.0, mass, 0.0,
            0.0, 0.0, mass,
        )
        diag[i] = diag[i] + mass_block


@wp.kernel
def _apply_velocity_update(
    particle_qd: wp.array(dtype=wp.vec3),
    dv: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    # output
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    """Update velocity: v_{n+1} = v_n + dv."""
    i = wp.tid()
    if particle_inv_mass[i] > 0.0:
        particle_qd_out[i] = particle_qd[i] + dv[i]
    else:
        particle_qd_out[i] = wp.vec3(0.0)


@wp.kernel
def _apply_position_update(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    dt: float,
    # output
    particle_q_out: wp.array(dtype=wp.vec3),
):
    """Update position: x_{n+1} = x_n + dt * v_{n+1}."""
    i = wp.tid()
    if particle_inv_mass[i] > 0.0:
        particle_q_out[i] = particle_q[i] + dt * particle_qd[i]
    else:
        particle_q_out[i] = particle_q[i]


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------


class SolverFEMShell(SolverBase):
    """GPU-accelerated FEM thin shell solver for stiff thin materials.

    Uses St. Venant-Kirchhoff membrane energy on triangle elements with
    implicit Euler time integration solved via Preconditioned Conjugate Gradient.

    Args:
        model: Newton Model containing the cloth/shell mesh.
        young_modulus: Young's modulus E in Pascals (default: 1e9 = 1 GPa).
        poisson_ratio: Poisson's ratio ν (default: 0.3).
        thickness: Shell thickness h in meters (default: 0.001 = 1mm).
        cg_tol: CG solver relative tolerance (default: 1e-6).
        cg_max_iter: Maximum CG iterations (default: 200).
        damping: Rayleigh damping coefficient (default: 0.01).
    """

    def __init__(
        self,
        model,
        young_modulus: float = 1.0e9,
        poisson_ratio: float = 0.3,
        thickness: float = 0.001,
        cg_tol: float = 1e-6,
        cg_max_iter: int = 200,
        damping: float = 0.01,
    ):
        super().__init__(model)

        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.thickness = thickness
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self.damping = damping
        self.last_residual = float("inf")

        # Compute Lamé parameters from E, ν
        E = young_modulus
        nu = poisson_ratio
        self.mu = E / (2.0 * (1.0 + nu))
        self.lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Scale by thickness for membrane (plane stress)
        # Membrane stiffness = h * material stiffness
        self.membrane_mu = self.mu * thickness
        self.membrane_lmbda = self.lmbda * thickness

        # Pre-allocate work arrays
        n = model.particle_count
        t = model.tri_count
        device = model.device

        self._elastic_forces = wp.zeros(n, dtype=wp.vec3, device=device)
        self._rhs = wp.zeros(n, dtype=wp.vec3, device=device)
        self._dv = wp.zeros(n, dtype=wp.vec3, device=device)

        # Stiffness matrix triplets: 9 entries per triangle (3×3 vertex pairs)
        self._triplet_rows = wp.zeros(t * 9, dtype=wp.int32, device=device)
        self._triplet_cols = wp.zeros(t * 9, dtype=wp.int32, device=device)
        self._triplet_vals = wp.zeros(t * 9, dtype=wp.mat33, device=device)

        # Finite difference epsilon for stiffness computation
        self._fd_eps = 1.0e-7

        # Cache gravity as wp.vec3 (model.gravity is a wp.array)
        g = model.gravity.numpy().flatten()
        self._gravity = wp.vec3(float(g[0]), float(g[1]), float(g[2]))

    def step(self, state_in, state_out, control, contacts, dt):
        """Simulate one time step using implicit Euler + PCG.

        Args:
            state_in: Input state (positions, velocities).
            state_out: Output state (updated positions, velocities).
            control: Control input (unused in Phase A).
            contacts: Contact information (used for ground collision).
            dt: Time step in seconds.
        """
        model = self.model
        n = model.particle_count
        device = model.device

        # 1. Compute elastic forces at current position
        self._elastic_forces.zero_()
        if model.tri_count > 0:
            wp.launch(
                _compute_membrane_forces,
                dim=model.tri_count,
                inputs=[
                    state_in.particle_q,
                    model.tri_indices,
                    model.tri_poses,
                    model.tri_areas,
                    self.membrane_mu,
                    self.membrane_lmbda,
                ],
                outputs=[self._elastic_forces],
                device=device,
            )

        # 2. Compute stiffness matrix K (via finite differences)
        if model.tri_count > 0:
            wp.launch(
                _compute_membrane_stiffness_triplets,
                dim=model.tri_count,
                inputs=[
                    state_in.particle_q,
                    model.tri_indices,
                    model.tri_poses,
                    model.tri_areas,
                    self.membrane_mu,
                    self.membrane_lmbda,
                    self._fd_eps,
                ],
                outputs=[
                    self._triplet_rows,
                    self._triplet_cols,
                    self._triplet_vals,
                ],
                device=device,
            )

        # 3. Assemble system matrix A = M + dt²K
        K = wps.bsr_from_triplets(
            rows_of_blocks=n,
            cols_of_blocks=n,
            rows=self._triplet_rows,
            columns=self._triplet_cols,
            values=self._triplet_vals,
        )

        # Scale K by dt²
        wps.bsr_scale(K, dt * dt)

        # Add mass to diagonal: A = dt²K + M
        diag = wps.bsr_get_diag(K)
        wp.launch(
            _apply_mass_diagonal,
            dim=n,
            inputs=[model.particle_inv_mass, dt],
            outputs=[diag],
            device=device,
        )
        wps.bsr_set_diag(K, diag)

        # Add Rayleigh damping: A += dt * α * K  (stiffness-proportional)
        # For simplicity, add damping to diagonal
        if self.damping > 0:
            wps.bsr_scale(K, 1.0 + dt * self.damping)

        # 4. Compute RHS: b = dt * f_total
        self._rhs.zero_()
        wp.launch(
            _implicit_euler_rhs,
            dim=n,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                model.particle_inv_mass,
                self._elastic_forces,
                self._gravity,
                dt,
            ],
            outputs=[self._rhs],
            device=device,
        )

        # 5. Solve: A @ dv = b  via PCG
        self._dv.zero_()
        M_precond = preconditioner(K, ptype="diag")

        residuals = []

        def _callback(i, err, tol_reached):
            residuals.append(float(err))

        cg(
            A=K,
            b=self._rhs,
            x=self._dv,
            tol=self.cg_tol,
            maxiter=self.cg_max_iter,
            M=M_precond,
            callback=_callback,
            use_cuda_graph=False,  # safer for first impl
        )

        self.last_residual = residuals[-1] if residuals else float("inf")

        # 6. Update velocity: v_{n+1} = v_n + dv
        wp.launch(
            _apply_velocity_update,
            dim=n,
            inputs=[
                state_in.particle_qd,
                self._dv,
                model.particle_inv_mass,
                model.particle_flags,
            ],
            outputs=[state_out.particle_qd],
            device=device,
        )

        # 7. Update position: x_{n+1} = x_n + dt * v_{n+1}
        wp.launch(
            _apply_position_update,
            dim=n,
            inputs=[
                state_in.particle_q,
                state_out.particle_qd,
                model.particle_inv_mass,
                dt,
            ],
            outputs=[state_out.particle_q],
            device=device,
        )

        # 8. Integrate rigid bodies (if any)
        self.integrate_bodies(model, state_in, state_out, dt)
