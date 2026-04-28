# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FEM Thin Shell Solver — Phase A: Membrane energy with implicit Euler + PCG.

GPU-accelerated FEM thin shell solver for stiff thin materials (cardboard,
sheet metal, plastic). Uses St. Venant-Kirchhoff membrane energy on triangle
elements with implicit Euler time integration solved via Conjugate Gradient.

The analytic Hessian is adapted from Newton's VBD StVK implementation
(particle_vbd_kernels.py) but restructured for global matrix assembly.
"""

import numpy as np
import scipy.sparse
import warp as wp
import warp.sparse as wps
from warp.optim.linear import cg, preconditioner

try:
    import ipctk
    _HAS_IPCTK = True
except ImportError:
    _HAS_IPCTK = False

from newton._src.solvers.solver import SolverBase


# ---------------------------------------------------------------------------
# Analytic StVK: per-element force + stiffness (9x9 → nine 3x3 blocks)
# ---------------------------------------------------------------------------


@wp.func
def _stvk_vertex_force_and_hessian(
    v_order: int,
    f0: wp.vec3,
    f1: wp.vec3,
    area: float,
    mu: float,
    lmbda: float,
    DmInv00: float,
    DmInv01: float,
    DmInv10: float,
    DmInv11: float,
):
    """Compute StVK force and Hessian for one vertex of a triangle.

    Adapted from Newton VBD's evaluate_stvk_force_hessian.
    Returns (force_vec3, hessian_mat33) for vertex v_order.
    """
    # Green strain G = 0.5(F^T F - I)
    f0f0 = wp.dot(f0, f0)
    f1f1 = wp.dot(f1, f1)
    f0f1 = wp.dot(f0, f1)

    G00 = 0.5 * (f0f0 - 1.0)
    G11 = 0.5 * (f1f1 - 1.0)
    G01 = 0.5 * f0f1

    trace_G = G00 + G11

    # First Piola-Kirchhoff stress: PK1 = 2*mu*F*G + lambda*tr(G)*F
    lt = lmbda * trace_G
    two_mu = 2.0 * mu

    PK1_col0 = f0 * (two_mu * G00 + lt) + f1 * (two_mu * G01)
    PK1_col1 = f0 * (two_mu * G01) + f1 * (two_mu * G11 + lt)

    # dF/dx for this vertex
    mask0 = float(v_order == 0)
    mask1 = float(v_order == 1)
    mask2 = float(v_order == 2)

    df0_dx = DmInv00 * (mask1 - mask0) + DmInv10 * (mask2 - mask0)
    df1_dx = DmInv01 * (mask1 - mask0) + DmInv11 * (mask2 - mask0)

    # Force: f_i = -area * PK1 : dF/dx_i
    force = -area * (PK1_col0 * df0_dx + PK1_col1 * df1_dx)

    # Hessian: see VBD kernel for derivation
    Ic = f0f0 + f1f1
    two_dpsi_dIc = -mu + (0.5 * Ic - 1.0) * lmbda
    I33 = wp.identity(n=3, dtype=float)

    f0_o_f0 = wp.outer(f0, f0)
    f1_o_f1 = wp.outer(f1, f1)
    f0_o_f1 = wp.outer(f0, f1)
    f1_o_f0 = wp.outer(f1, f0)

    H00 = lmbda * f0_o_f0 + two_dpsi_dIc * I33 + mu * (f0f0 * I33 + 2.0 * f0_o_f0 + f1_o_f1)
    H01 = lmbda * f0_o_f1 + mu * (f0f1 * I33 + f1_o_f0)
    H11 = lmbda * f1_o_f1 + two_dpsi_dIc * I33 + mu * (f1f1 * I33 + 2.0 * f1_o_f1 + f0_o_f0)

    df0sq = df0_dx * df0_dx
    df1sq = df1_dx * df1_dx
    df01 = df0_dx * df1_dx

    hessian = area * (df0sq * H00 + df1sq * H11 + df01 * (H01 + wp.transpose(H01)))

    return force, hessian


@wp.kernel
def _compute_element_stiffness(
    particle_q: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=wp.int32),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_areas: wp.array(dtype=float),
    mu: float,
    lmbda: float,
    # outputs: forces + stiffness triplets (9 blocks per triangle)
    forces: wp.array(dtype=wp.vec3),
    triplet_rows: wp.array(dtype=wp.int32),
    triplet_cols: wp.array(dtype=wp.int32),
    triplet_vals: wp.array(dtype=wp.mat33),
):
    """Compute analytic forces and stiffness matrix entries for each triangle.

    For each triangle, computes:
    - 3 vertex forces (atomic_add to global force array)
    - 9 stiffness blocks K[vi,vj] (written as triplets for BSR assembly)

    The stiffness block K[vi,vj] = d(force_vi)/d(pos_vj) computed analytically.
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

    DmInv00 = Dm_inv[0, 0]
    DmInv01 = Dm_inv[0, 1]
    DmInv10 = Dm_inv[1, 0]
    DmInv11 = Dm_inv[1, 1]

    # Deformation gradient columns
    e1 = p1 - p0
    e2 = p2 - p0

    # Check for degenerate triangle (current area vs rest area)
    current_area_2x = wp.length(wp.cross(e1, e2))
    if current_area_2x < area * 0.01:  # triangle compressed to <1% of rest area
        for vi in range(3):
            for vj in range(3):
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
                triplet_vals[out_idx] = wp.mat33(0.0)
        return

    f0 = e1 * DmInv00 + e2 * DmInv10
    f1 = e1 * DmInv01 + e2 * DmInv11

    # Green strain for energy check
    G_frob_sq = 0.0
    f0f0 = wp.dot(f0, f0)
    f1f1 = wp.dot(f1, f1)
    f0f1 = wp.dot(f0, f1)
    G00 = 0.5 * (f0f0 - 1.0)
    G11 = 0.5 * (f1f1 - 1.0)
    G01 = 0.5 * f0f1
    G_frob_sq = G00 * G00 + G11 * G11 + 2.0 * G01 * G01

    # Skip nearly-undeformed triangles (avoid numerical noise)
    if G_frob_sq < 1.0e-20:
        for vi in range(3):
            for vj in range(3):
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
                triplet_vals[out_idx] = wp.mat33(0.0)
        return

    # Compute PK1 stress for forces
    trace_G = G00 + G11
    lt = lmbda * trace_G
    two_mu = 2.0 * mu

    PK1_col0 = f0 * (two_mu * G00 + lt) + f1 * (two_mu * G01)
    PK1_col1 = f0 * (two_mu * G01) + f1 * (two_mu * G11 + lt)

    # Precompute Hessian blocks (independent of vertex)
    Ic = f0f0 + f1f1
    two_dpsi_dIc = -mu + (0.5 * Ic - 1.0) * lmbda
    I33 = wp.identity(n=3, dtype=float)

    f0_o_f0 = wp.outer(f0, f0)
    f1_o_f1 = wp.outer(f1, f1)
    f0_o_f1 = wp.outer(f0, f1)
    f1_o_f0 = wp.outer(f1, f0)

    d2E_dF2_00 = lmbda * f0_o_f0 + two_dpsi_dIc * I33 + mu * (f0f0 * I33 + 2.0 * f0_o_f0 + f1_o_f1)
    d2E_dF2_01 = lmbda * f0_o_f1 + mu * (f0f1 * I33 + f1_o_f0)
    d2E_dF2_11 = lmbda * f1_o_f1 + two_dpsi_dIc * I33 + mu * (f1f1 * I33 + 2.0 * f1_o_f1 + f0_o_f0)
    d2E_dF2_01T = wp.transpose(d2E_dF2_01)

    for vi in range(3):
        # dF/dx for vertex vi
        mask0_i = float(vi == 0)
        mask1_i = float(vi == 1)
        mask2_i = float(vi == 2)
        df0_dxi = DmInv00 * (mask1_i - mask0_i) + DmInv10 * (mask2_i - mask0_i)
        df1_dxi = DmInv01 * (mask1_i - mask0_i) + DmInv11 * (mask2_i - mask0_i)

        # Force on vertex vi
        force_vi = -area * (PK1_col0 * df0_dxi + PK1_col1 * df1_dxi)

        if vi == 0:
            wp.atomic_add(forces, v0, force_vi)
        elif vi == 1:
            wp.atomic_add(forces, v1, force_vi)
        else:
            wp.atomic_add(forces, v2, force_vi)

        for vj in range(3):
            # dF/dx for vertex vj
            mask0_j = float(vj == 0)
            mask1_j = float(vj == 1)
            mask2_j = float(vj == 2)
            df0_dxj = DmInv00 * (mask1_j - mask0_j) + DmInv10 * (mask2_j - mask0_j)
            df1_dxj = DmInv01 * (mask1_j - mask0_j) + DmInv11 * (mask2_j - mask0_j)

            # K[vi,vj] = area * (dF/dx_i)^T d2E/dF2 (dF/dx_j)
            K_block = area * (
                df0_dxi * df0_dxj * d2E_dF2_00
                + df1_dxi * df1_dxj * d2E_dF2_11
                + df0_dxi * df1_dxj * d2E_dF2_01
                + df1_dxi * df0_dxj * d2E_dF2_01T
            )

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


@wp.kernel
def _compute_bending_forces_and_stiffness(
    particle_q: wp.array(dtype=wp.vec3),
    edge_indices: wp.array2d(dtype=wp.int32),
    edge_rest_angle: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    bending_stiffness: float,
    # outputs: forces + stiffness triplets (16 blocks per edge: 4x4 vertex pairs)
    forces: wp.array(dtype=wp.vec3),
    bend_triplet_rows: wp.array(dtype=wp.int32),
    bend_triplet_cols: wp.array(dtype=wp.int32),
    bend_triplet_vals: wp.array(dtype=wp.mat33),
):
    """Compute bending forces and stiffness via dihedral angle energy.

    Bending energy per edge: W = stiffness * (theta - theta_0)^2 * L / 2
    where theta is the dihedral angle, theta_0 is rest angle, L is edge length.

    Edge indices: [opp0, opp1, shared0, shared1]
    Skip boundary edges (opp0 == -1 or opp1 == -1).

    Stiffness is computed via finite differences of the bending force
    (analytic bending Hessian is complex; FD is acceptable for Phase B POC
    since bending stiffness is much smaller than membrane stiffness).
    """
    eid = wp.tid()

    i0 = edge_indices[eid, 0]  # opposite vertex 0
    i1 = edge_indices[eid, 1]  # opposite vertex 1
    i2 = edge_indices[eid, 2]  # shared vertex 0 (edge start)
    i3 = edge_indices[eid, 3]  # shared vertex 1 (edge end)

    # Skip boundary edges
    if i0 == -1 or i1 == -1:
        for vi in range(4):
            for vj in range(4):
                out_idx = eid * 16 + vi * 4 + vj
                bend_triplet_rows[out_idx] = 0
                bend_triplet_cols[out_idx] = 0
                bend_triplet_vals[out_idx] = wp.mat33(0.0)
        return

    p0 = particle_q[i0]
    p1 = particle_q[i1]
    p2 = particle_q[i2]
    p3 = particle_q[i3]

    rest_angle = edge_rest_angle[eid]
    rest_len = edge_rest_length[eid]

    # Compute dihedral angle
    e = p3 - p2  # shared edge vector
    e_len = wp.length(e)
    if e_len < 1.0e-10:
        for vi in range(4):
            for vj in range(4):
                out_idx = eid * 16 + vi * 4 + vj
                bend_triplet_rows[out_idx] = 0
                bend_triplet_cols[out_idx] = 0
                bend_triplet_vals[out_idx] = wp.mat33(0.0)
        return

    e_hat = e / e_len

    # Face normals
    n0 = wp.cross(e, p0 - p2)
    n1 = wp.cross(p1 - p2, e)

    n0_len = wp.length(n0)
    n1_len = wp.length(n1)

    if n0_len < 1.0e-10 or n1_len < 1.0e-10:
        for vi in range(4):
            for vj in range(4):
                out_idx = eid * 16 + vi * 4 + vj
                bend_triplet_rows[out_idx] = 0
                bend_triplet_cols[out_idx] = 0
                bend_triplet_vals[out_idx] = wp.mat33(0.0)
        return

    n0_hat = n0 / n0_len
    n1_hat = n1 / n1_len

    cos_theta = wp.dot(n0_hat, n1_hat)
    cos_theta = wp.clamp(cos_theta, -1.0, 1.0)
    sin_theta = wp.dot(wp.cross(n0_hat, n1_hat), e_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    # Bending energy (Discrete Shells, Grinspun 2003):
    # W = kappa_e * (theta - theta_0)^2, kappa_e = 3*D/|ē|
    # where D = E*h³/(12*(1-ν²)) is the flexural rigidity
    # Force: f = -dW/dx = -2*kappa_e*(theta-theta_0)*dtheta/dx
    delta_theta = theta - rest_angle
    kappa_e = 3.0 * bending_stiffness / rest_len

    # Skip if bending stiffness is negligible
    if kappa_e < 1.0e-6:
        for vi in range(4):
            for vj in range(4):
                out_idx = eid * 16 + vi * 4 + vj
                bend_triplet_rows[out_idx] = 0
                bend_triplet_cols[out_idx] = 0
                bend_triplet_vals[out_idx] = wp.mat33(0.0)
        return

    # Heights from edge to opposite vertices (for gradient computation)
    h0 = n0_len / e_len  # distance from opp0 to edge
    h1 = n1_len / e_len  # distance from opp1 to edge

    # Bending force gradients (dtheta/dx for each vertex)
    # See Grinspun et al. "Discrete Shells" for derivation
    # Signs verified numerically: moving opp0 in n0 direction DECREASES theta
    grad0 = -n0_hat / h0  # dtheta/dx0 (opposite vertex of face 0)
    grad1 = -n1_hat / h1  # dtheta/dx1 (opposite vertex of face 1)

    # For shared vertices, use chain rule with edge parametric coords
    t02 = wp.dot(p0 - p2, e_hat) / e_len  # parametric coord of p0 projected onto edge
    t12 = wp.dot(p1 - p2, e_hat) / e_len

    grad2 = -(1.0 - t02) * grad0 - (1.0 - t12) * grad1  # dtheta/dx2 (edge start)
    grad3 = -t02 * grad0 - t12 * grad1  # dtheta/dx3 (edge end) -- note: signs from discrete shells
    # Correction: grad2 + grad3 = -(grad0 + grad1) for force balance
    grad3 = -(grad0 + grad1 + grad2)

    # Force = -dW/dx = -2 * kappa_e * delta_theta * dtheta/dx
    f_scale = -2.0 * kappa_e * delta_theta

    wp.atomic_add(forces, i0, f_scale * grad0)
    wp.atomic_add(forces, i1, f_scale * grad1)
    wp.atomic_add(forces, i2, f_scale * grad2)
    wp.atomic_add(forces, i3, f_scale * grad3)

    # Stiffness blocks: K[vi,vj] = 2 * kappa_e * grad_i ⊗ grad_j
    # (rank-1 Hessian approximation of W = kappa_e * (delta_theta)^2)
    # Non-zero at rest is correct: provides resistance to any perturbation
    # from the rest angle.
    stiffness_scale = 2.0 * kappa_e
    grads_val_0 = grad0
    grads_val_1 = grad1
    grads_val_2 = grad2
    grads_val_3 = grad3

    vertex_ids_0 = i0
    vertex_ids_1 = i1
    vertex_ids_2 = i2
    vertex_ids_3 = i3

    for vi in range(4):
        if vi == 0:
            gi = grads_val_0
            ri = vertex_ids_0
        elif vi == 1:
            gi = grads_val_1
            ri = vertex_ids_1
        elif vi == 2:
            gi = grads_val_2
            ri = vertex_ids_2
        else:
            gi = grads_val_3
            ri = vertex_ids_3

        for vj in range(4):
            if vj == 0:
                gj = grads_val_0
                cj = vertex_ids_0
            elif vj == 1:
                gj = grads_val_1
                cj = vertex_ids_1
            elif vj == 2:
                gj = grads_val_2
                cj = vertex_ids_2
            else:
                gj = grads_val_3
                cj = vertex_ids_3

            K_block = stiffness_scale * wp.outer(gi, gj)

            out_idx = eid * 16 + vi * 4 + vj
            bend_triplet_rows[out_idx] = ri
            bend_triplet_cols[out_idx] = cj
            bend_triplet_vals[out_idx] = K_block


@wp.kernel
def _add_external_forces(
    external_f: wp.array(dtype=wp.vec3),
    elastic_f: wp.array(dtype=wp.vec3),
    out_f: wp.array(dtype=wp.vec3),
):
    """Add external/contact forces to elastic forces."""
    i = wp.tid()
    out_f[i] = elastic_f[i] + external_f[i]


@wp.kernel
def _compute_simple_contacts(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    ground_z: float,
    sphere_center: wp.vec3,
    sphere_radius: float,
    contact_stiffness: float,
    contact_damping: float,
    particle_qd: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
):
    """Ground + sphere penalty contact with cubic stiffness."""
    i = wp.tid()
    if particle_inv_mass[i] <= 0.0:
        return
    p = particle_q[i]
    v = particle_qd[i]
    margin = 0.005
    # Ground — cubic penalty
    depth = ground_z + margin - p[2]
    if depth > 0.0:
        f_z = contact_stiffness * depth * depth * depth + contact_damping * wp.max(-v[2], 0.0)
        wp.atomic_add(forces, i, wp.vec3(0.0, 0.0, f_z))
    # Sphere — cubic penalty
    to_s = p - sphere_center
    dist = wp.length(to_s)
    pen = (sphere_radius + margin) - dist
    if pen > 0.0 and dist > 1.0e-8:
        n = to_s / dist
        vn = wp.dot(v, n)
        fc = contact_stiffness * pen * pen * pen + contact_damping * wp.max(-vn, 0.0)
        wp.atomic_add(forces, i, n * fc)


@wp.kernel
def _project_contacts(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    ground_z: float,
    sphere_center: wp.vec3,
    sphere_radius: float,
):
    """Project particles out of collision volumes (position correction)."""
    i = wp.tid()
    if particle_inv_mass[i] <= 0.0:
        return
    p = particle_q[i]

    # Ground projection
    if p[2] < ground_z:
        particle_q[i] = wp.vec3(p[0], p[1], ground_z)
        v = particle_qd[i]
        if v[2] < 0.0:
            particle_qd[i] = wp.vec3(v[0], v[1], 0.0)

    # Sphere projection
    p = particle_q[i]  # re-read after ground fix
    to_s = p - sphere_center
    dist = wp.length(to_s)
    if dist < sphere_radius and dist > 1.0e-8:
        n = to_s / dist
        particle_q[i] = sphere_center + n * sphere_radius
        v = particle_qd[i]
        vn = wp.dot(v, n)
        if vn < 0.0:
            particle_qd[i] = v - n * vn  # remove inward velocity


@wp.kernel
def _update_plastic_rest_angles(
    particle_q: wp.array(dtype=wp.vec3),
    edge_indices: wp.array2d(dtype=wp.int32),
    edge_rest_angle: wp.array(dtype=float),
    yield_angle: float,
    plasticity_rate: float,
):
    """Update rest angles for plastic deformation.

    When the current dihedral angle deviates from rest by more than
    yield_angle, the rest angle creeps toward the current angle.
    This creates permanent creases.
    """
    eid = wp.tid()
    i0 = edge_indices[eid, 0]
    i1 = edge_indices[eid, 1]
    i2 = edge_indices[eid, 2]
    i3 = edge_indices[eid, 3]

    if i0 == -1 or i1 == -1:
        return

    p0 = particle_q[i0]
    p1 = particle_q[i1]
    p2 = particle_q[i2]
    p3 = particle_q[i3]

    # Compute current dihedral angle
    e = p3 - p2
    e_len = wp.length(e)
    if e_len < 1.0e-10:
        return
    e_hat = e / e_len

    n0 = wp.cross(e, p0 - p2)
    n1 = wp.cross(p1 - p2, e)
    n0_len = wp.length(n0)
    n1_len = wp.length(n1)
    if n0_len < 1.0e-10 or n1_len < 1.0e-10:
        return

    n0_hat = n0 / n0_len
    n1_hat = n1 / n1_len
    cos_theta = wp.clamp(wp.dot(n0_hat, n1_hat), -1.0, 1.0)
    sin_theta = wp.dot(wp.cross(n0_hat, n1_hat), e_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    rest = edge_rest_angle[eid]
    delta = theta - rest

    # If deformation exceeds yield, update rest angle
    if wp.abs(delta) > yield_angle:
        # Creep rest angle toward current (permanent deformation)
        sign = 1.0
        if delta < 0.0:
            sign = -1.0
        excess = wp.abs(delta) - yield_angle
        edge_rest_angle[eid] = rest + sign * excess * plasticity_rate


@wp.kernel
def _implicit_euler_rhs(
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    elastic_forces: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
    rhs: wp.array(dtype=wp.vec3),
):
    """RHS of implicit Euler: b = dt*(f_elastic + f_gravity) + M*v_n.

    System: (M + dt²K) dv = dt*f + M*(0) → simplified as:
    (M + dt²K) dv = dt * f_total
    v_{n+1} = v_n + dv
    x_{n+1} = x_n + dt * v_{n+1}
    """
    i = wp.tid()
    inv_m = particle_inv_mass[i]
    if inv_m > 0.0:
        mass = 1.0 / inv_m
        f_grav = gravity * mass
        rhs[i] = dt * (elastic_forces[i] + f_grav)
    else:
        rhs[i] = wp.vec3(0.0)


@wp.kernel
def _add_mass_to_diagonal(
    particle_inv_mass: wp.array(dtype=float),
    diag: wp.array(dtype=wp.mat33),
):
    """Add mass matrix to diagonal: A_ii += m_i * I."""
    i = wp.tid()
    inv_m = particle_inv_mass[i]
    if inv_m > 0.0:
        mass = 1.0 / inv_m
        diag[i] = diag[i] + wp.mat33(
            mass, 0.0, 0.0,
            0.0, mass, 0.0,
            0.0, 0.0, mass,
        )


@wp.kernel
def _update_velocity(
    particle_qd_in: wp.array(dtype=wp.vec3),
    dv: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    damping: float,
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    """v_{n+1} = (1 - damping) * (v_n + dv), clamped to max velocity."""
    i = wp.tid()
    if particle_inv_mass[i] > 0.0:
        new_v = (1.0 - damping) * (particle_qd_in[i] + dv[i])
        # Clamp velocity to prevent explosion
        speed = wp.length(new_v)
        max_speed = 5.0  # m/s — lower clamp prevents energy buildup
        if speed > max_speed:
            new_v = new_v * (max_speed / speed)
        particle_qd_out[i] = new_v
    else:
        particle_qd_out[i] = wp.vec3(0.0)


@wp.kernel
def _update_position(
    particle_q_in: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    dt: float,
    particle_q_out: wp.array(dtype=wp.vec3),
):
    """x_{n+1} = x_n + dt * v_{n+1}."""
    i = wp.tid()
    if particle_inv_mass[i] > 0.0:
        particle_q_out[i] = particle_q_in[i] + dt * particle_qd[i]
    else:
        particle_q_out[i] = particle_q_in[i]


# ---------------------------------------------------------------------------
# Solver
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
        damping: Velocity damping per step (default: 0.005).
    """

    def __init__(
        self,
        model,
        young_modulus: float = 1.0e9,
        poisson_ratio: float = 0.3,
        thickness: float = 0.001,
        cg_tol: float = 1e-6,
        cg_max_iter: int = 200,
        damping: float = 0.005,
        substeps: int = 8,
        yield_angle: float = 0.0,
        plasticity_rate: float = 0.5,
        use_ipc: bool = False,
        ipc_dhat: float = 0.01,
    ):
        super().__init__(model)

        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.thickness = thickness
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self.damping = damping
        self.substeps = substeps
        self.yield_angle = yield_angle
        self.plasticity_rate = plasticity_rate
        self.last_residual = float("inf")

        # IPC contact
        self.use_ipc = use_ipc
        self.ipc_dhat = ipc_dhat
        self._ipc_ground_z = None
        self._ipc_ground_normal = None
        self._ipc_ground_origin = None
        self._ipc_friction_mu = 0.0
        self._ipc_kappa = None  # barrier stiffness, auto-estimated on first step
        self._ipc_collision_mesh = None
        self._ipc_prev_min_dist = None

        # IPC sphere (unified mesh approach)
        self._ipc_sphere_center = None
        self._ipc_sphere_radius = None
        self._ipc_sphere_friction = 0.0
        self._ipc_sphere_verts = None  # (n_sphere, 3)
        self._ipc_sphere_faces = None  # (t_sphere, 3)
        self._ipc_unified_collision_mesh = None  # rebuilt when sphere moves

        if use_ipc and not _HAS_IPCTK:
            raise ImportError("use_ipc=True requires ipctk: pip install ipctk")

        # Lamé parameters from E, ν — plane STRESS for thin shells
        # (not plane strain which has 1-2ν denominator and diverges as ν→0.5)
        E = young_modulus
        nu = poisson_ratio
        self.mu = E / (2.0 * (1.0 + nu))
        self.lmbda = E * nu / ((1.0 + nu) * (1.0 - nu))

        # Scale by thickness for membrane energy
        self.membrane_mu = self.mu * thickness
        self.membrane_lmbda = self.lmbda * thickness

        # Bending stiffness: κ = E*h³/(12*(1-ν²)) (Kirchhoff plate theory)
        self.bending_stiffness = young_modulus * thickness**3 / (12.0 * (1.0 - poisson_ratio**2))

        # Pre-allocate work arrays
        n = model.particle_count
        t = model.tri_count
        e = model.edge_count
        device = model.device

        self._elastic_forces = wp.zeros(n, dtype=wp.vec3, device=device)
        self._rhs = wp.zeros(n, dtype=wp.vec3, device=device)
        self._dv = wp.zeros(n, dtype=wp.vec3, device=device)

        # Stiffness triplets: 9 entries per triangle
        self._triplet_rows = wp.zeros(t * 9, dtype=wp.int32, device=device)
        self._triplet_cols = wp.zeros(t * 9, dtype=wp.int32, device=device)
        self._triplet_vals = wp.zeros(t * 9, dtype=wp.mat33, device=device)

        # Bending stiffness triplets: 16 entries per edge (4×4 vertex pairs)
        self._bend_triplet_rows = wp.zeros(e * 16, dtype=wp.int32, device=device)
        self._bend_triplet_cols = wp.zeros(e * 16, dtype=wp.int32, device=device)
        self._bend_triplet_vals = wp.zeros(e * 16, dtype=wp.mat33, device=device)

        # Cache gravity
        g = model.gravity.numpy().flatten()
        self._gravity = wp.vec3(float(g[0]), float(g[1]), float(g[2]))

        # Temp buffer for position update (avoid read/write race)
        self._q_temp = wp.zeros(n, dtype=wp.vec3, device=device)

    def step(self, state_in, state_out, control, contacts, dt):
        """Simulate one time step using substeps with single-step implicit Euler.

        Each substep solves:
            (M + dt²K) dv = dt * f(x_n) + dt² * K * v_n
            v_{n+1} = v_n + dv
            x_{n+1} = x_n + dt * v_{n+1}

        Single linearization per substep (no NR iterations). Stability comes
        from small enough substeps relative to stiffness.
        """
        model = self.model
        n = model.particle_count
        device = model.device
        sub_dt = dt / self.substeps

        # Copy input state
        wp.copy(state_out.particle_q, state_in.particle_q)
        wp.copy(state_out.particle_qd, state_in.particle_qd)

        for _sub in range(self.substeps):
            # TODO Phase C: proper IPC contact. Newton's penalty contacts
            # are too stiff for implicit Euler — skip for now.

            # 1. Compute elastic forces + stiffness at current position
            self._elastic_forces.zero_()
            self._triplet_vals.zero_()
            self._bend_triplet_vals.zero_()

            if model.tri_count > 0:
                wp.launch(
                    _compute_element_stiffness,
                    dim=model.tri_count,
                    inputs=[
                        state_out.particle_q,
                        model.tri_indices, model.tri_poses, model.tri_areas,
                        self.membrane_mu, self.membrane_lmbda,
                    ],
                    outputs=[
                        self._elastic_forces,
                        self._triplet_rows, self._triplet_cols, self._triplet_vals,
                    ],
                    device=device,
                )

            # 1b. Compute bending forces + stiffness
            if model.edge_count > 0:
                wp.launch(
                    _compute_bending_forces_and_stiffness,
                    dim=model.edge_count,
                    inputs=[
                        state_out.particle_q,
                        model.edge_indices,
                        model.edge_rest_angle,
                        model.edge_rest_length,
                        self.bending_stiffness,
                    ],
                    outputs=[
                        self._elastic_forces,
                        self._bend_triplet_rows,
                        self._bend_triplet_cols,
                        self._bend_triplet_vals,
                    ],
                    device=device,
                )

            # 1c. Contact forces
            ipc_grad = None
            ipc_hess_diag = None
            ipc_self_grad = None
            ipc_self_hess = None

            if self.use_ipc:
                # Transfer positions GPU → CPU for ipctk
                wp.synchronize()
                V_np = state_out.particle_q.numpy().astype(np.float64)

                # Ground plane IPC barrier
                ipc_grad, ipc_hess_diag = self._compute_ipc_contact(V_np)

                # Self-collision IPC barrier
                ipc_self_grad, ipc_self_hess = self._compute_ipc_self_collision(V_np)

                # Add IPC ground forces to elastic forces on GPU
                ipc_forces_wp = wp.array(ipc_grad.astype(np.float32), dtype=wp.vec3, device=device)
                wp.launch(
                    _add_external_forces,
                    dim=n,
                    inputs=[ipc_forces_wp, self._elastic_forces],
                    outputs=[self._elastic_forces],
                    device=device,
                )

                # Add self-collision forces
                if ipc_self_grad is not None:
                    self_forces = ipc_self_grad.reshape(-1, 3).astype(np.float32)
                    self_forces_wp = wp.array(self_forces, dtype=wp.vec3, device=device)
                    wp.launch(
                        _add_external_forces,
                        dim=n,
                        inputs=[self_forces_wp, self._elastic_forces],
                        outputs=[self._elastic_forces],
                        device=device,
                    )

            elif hasattr(self, '_contact_sphere_center'):
                wp.launch(
                    _compute_simple_contacts,
                    dim=n,
                    inputs=[
                        state_out.particle_q,
                        model.particle_inv_mass,
                        0.0,  # ground_z
                        self._contact_sphere_center,
                        self._contact_sphere_radius,
                        self._contact_stiffness,
                        self._contact_damping,
                        state_out.particle_qd,
                    ],
                    outputs=[self._elastic_forces],
                    device=device,
                )

            # 2. Assemble A = M + dt²(K_membrane + K_bending)
            # Pre-allocate combined triplet arrays (avoid numpy concat in hot loop)
            n_mem = model.tri_count * 9
            n_bend = model.edge_count * 16

            # Count IPC contact triplets
            n_ipc_ground = 0
            n_ipc_self = 0
            ipc_ground_rows = None
            ipc_ground_cols = None
            ipc_ground_vals = None
            ipc_self_rows = None
            ipc_self_cols = None
            ipc_self_vals = None

            if self.use_ipc and ipc_hess_diag is not None:
                # Ground plane: diagonal blocks only (n blocks)
                nz_mask = np.any(np.abs(ipc_hess_diag) > 1e-20, axis=(1, 2))
                nz_indices = np.where(nz_mask)[0]
                n_ipc_ground = len(nz_indices)
                if n_ipc_ground > 0:
                    ipc_ground_rows = wp.array(nz_indices.astype(np.int32), dtype=wp.int32, device=device)
                    ipc_ground_cols = wp.array(nz_indices.astype(np.int32), dtype=wp.int32, device=device)
                    # Scale by dt^2 to match the stiffness matrix scaling
                    scaled_vals = (ipc_hess_diag[nz_indices] * sub_dt * sub_dt).astype(np.float32)
                    ipc_ground_vals = wp.array(
                        scaled_vals.reshape(-1, 3, 3),
                        dtype=wp.mat33, device=device
                    )

            if self.use_ipc and ipc_self_hess is not None:
                # Self-collision: scipy CSR sparse → triplets
                # Vectorized CSR → BSR conversion (no Python loop — #51)
                coo = ipc_self_hess.tocoo()
                block_rows = (coo.row // 3).astype(np.int32)
                block_cols = (coo.col // 3).astype(np.int32)
                local_r = coo.row % 3
                local_c = coo.col % 3
                # Unique block keys and vectorized accumulation
                block_key = block_rows.astype(np.int64) * n + block_cols.astype(np.int64)
                unique_keys, inverse = np.unique(block_key, return_inverse=True)
                n_ipc_self = len(unique_keys)
                if n_ipc_self > 0:
                    self_block_rows = (unique_keys // n).astype(np.int32)
                    self_block_cols = (unique_keys % n).astype(np.int32)
                    # Vectorized: encode (block_idx, local_r, local_c) → flat index
                    flat_idx = inverse * 9 + local_r * 3 + local_c
                    self_block_vals_flat = np.zeros(n_ipc_self * 9, dtype=np.float64)
                    np.add.at(self_block_vals_flat, flat_idx, coo.data)
                    self_block_vals = (self_block_vals_flat.reshape(n_ipc_self, 3, 3) * sub_dt * sub_dt).astype(np.float32)
                    ipc_self_rows = wp.array(self_block_rows, dtype=wp.int32, device=device)
                    ipc_self_cols = wp.array(self_block_cols, dtype=wp.int32, device=device)
                    ipc_self_vals = wp.array(self_block_vals, dtype=wp.mat33, device=device)

            n_total = n_mem + n_bend + n_ipc_ground + n_ipc_self

            if not hasattr(self, '_all_rows') or self._all_rows.shape[0] != n_total:
                self._all_rows = wp.zeros(n_total, dtype=wp.int32, device=device)
                self._all_cols = wp.zeros(n_total, dtype=wp.int32, device=device)
                self._all_vals = wp.zeros(n_total, dtype=wp.mat33, device=device)

            # Copy membrane triplets to combined arrays
            wp.copy(self._all_rows, self._triplet_rows, dest_offset=0, src_offset=0, count=n_mem)
            wp.copy(self._all_cols, self._triplet_cols, dest_offset=0, src_offset=0, count=n_mem)
            wp.copy(self._all_vals, self._triplet_vals, dest_offset=0, src_offset=0, count=n_mem)
            # Copy bending triplets
            wp.copy(self._all_rows, self._bend_triplet_rows, dest_offset=n_mem, src_offset=0, count=n_bend)
            wp.copy(self._all_cols, self._bend_triplet_cols, dest_offset=n_mem, src_offset=0, count=n_bend)
            wp.copy(self._all_vals, self._bend_triplet_vals, dest_offset=n_mem, src_offset=0, count=n_bend)
            # Copy IPC ground triplets
            offset = n_mem + n_bend
            if n_ipc_ground > 0:
                wp.copy(self._all_rows, ipc_ground_rows, dest_offset=offset, src_offset=0, count=n_ipc_ground)
                wp.copy(self._all_cols, ipc_ground_cols, dest_offset=offset, src_offset=0, count=n_ipc_ground)
                wp.copy(self._all_vals, ipc_ground_vals, dest_offset=offset, src_offset=0, count=n_ipc_ground)
                offset += n_ipc_ground
            # Copy IPC self-collision triplets
            if n_ipc_self > 0:
                wp.copy(self._all_rows, ipc_self_rows, dest_offset=offset, src_offset=0, count=n_ipc_self)
                wp.copy(self._all_cols, ipc_self_cols, dest_offset=offset, src_offset=0, count=n_ipc_self)
                wp.copy(self._all_vals, ipc_self_vals, dest_offset=offset, src_offset=0, count=n_ipc_self)

            K = wps.bsr_from_triplets(
                rows_of_blocks=n, cols_of_blocks=n,
                rows=self._all_rows, columns=self._all_cols,
                values=self._all_vals,
            )
            wps.bsr_scale(K, sub_dt * sub_dt)

            diag = wps.bsr_get_diag(K)
            wp.launch(
                _add_mass_to_diagonal,
                dim=n,
                inputs=[model.particle_inv_mass],
                outputs=[diag],
                device=device,
            )
            wps.bsr_set_diag(K, diag)

            # 3. RHS: b = dt * (f_elastic + f_gravity)
            self._rhs.zero_()
            wp.launch(
                _implicit_euler_rhs,
                dim=n,
                inputs=[
                    state_out.particle_qd, model.particle_inv_mass,
                    self._elastic_forces, self._gravity, sub_dt,
                ],
                outputs=[self._rhs],
                device=device,
            )

            # 4. Solve (M + dt²K) dv = b
            self._dv.zero_()
            M_precond = preconditioner(K, ptype="diag")

            residuals = []
            def _callback(i, err, tol_reached):
                residuals.append(float(err))

            cg(A=K, b=self._rhs, x=self._dv,
               tol=self.cg_tol, maxiter=self.cg_max_iter,
               M=M_precond, callback=_callback, use_cuda_graph=False)

            self.last_residual = residuals[-1] if residuals else float("inf")

            # 5. Update velocity: v_{n+1} = (1-d)*(v_n + dv)
            wp.launch(
                _update_velocity,
                dim=n,
                inputs=[state_out.particle_qd, self._dv,
                        model.particle_inv_mass, self.damping],
                outputs=[state_out.particle_qd],
                device=device,
            )

            # 6. Update position: x_{n+1} = x_n + dt * v_{n+1}
            # With IPC: limit step via CCD to prevent tunneling
            if self.use_ipc:
                wp.synchronize()
                V_current = state_out.particle_q.numpy().astype(np.float64)
                V_vel = state_out.particle_qd.numpy().astype(np.float64)
                inv_mass = model.particle_inv_mass.numpy()
                # Build candidate positions
                V_candidate = V_current.copy()
                for i in range(n):
                    if inv_mass[i] > 0:
                        V_candidate[i] += sub_dt * V_vel[i]
                # CCD step size
                alpha = self._ipc_ccd_step_size(V_current, V_candidate)
                # Apply clamped step
                effective_dt = sub_dt * min(alpha, 1.0)
                # Scale velocity for this substep to match clamped step
                if alpha < 1.0:
                    scale = float(alpha)
                    vel_scaled = V_vel * scale
                    state_out.particle_qd = wp.array(
                        vel_scaled.astype(np.float32), dtype=wp.vec3, device=device
                    )

            wp.launch(
                _update_position,
                dim=n,
                inputs=[state_out.particle_q, state_out.particle_qd,
                        model.particle_inv_mass, sub_dt],
                outputs=[self._q_temp],
                device=device,
            )
            wp.copy(state_out.particle_q, self._q_temp)

            # Position projection: push particles out of colliders
            # (only for legacy penalty contact; IPC handles this via barrier + CCD)
            if not self.use_ipc and hasattr(self, '_contact_sphere_center'):
                wp.launch(
                    _project_contacts,
                    dim=n,
                    inputs=[
                        state_out.particle_q,
                        state_out.particle_qd,
                        model.particle_inv_mass,
                        0.0,  # ground_z
                        self._contact_sphere_center,
                        self._contact_sphere_radius,
                    ],
                    device=device,
                )

            # Plastic deformation: update rest angles if yield exceeded
            if self.yield_angle > 0.0 and model.edge_count > 0:
                wp.launch(
                    _update_plastic_rest_angles,
                    dim=model.edge_count,
                    inputs=[
                        state_out.particle_q,
                        model.edge_indices,
                        model.edge_rest_angle,
                        self.yield_angle,
                        self.plasticity_rate,
                    ],
                    device=device,
                )

            # IPC friction via ipctk TangentialCollisions (#49)
            mu = max(self._ipc_friction_mu, self._ipc_sphere_friction)
            if self.use_ipc and mu > 0:
                wp.synchronize()
                pos_np = state_out.particle_q.numpy().astype(np.float64)
                vel_np = state_out.particle_qd.numpy().astype(np.float64)

                # Get previous positions for velocity estimation
                if not hasattr(self, '_ipc_prev_V') or self._ipc_prev_V is None:
                    self._ipc_prev_V = pos_np.copy()

                try:
                    cm, n_shell = self._build_ipc_collision_mesh()
                    # Build unified V if sphere present
                    if self._ipc_sphere_center is not None:
                        sphere_V = self._ipc_sphere_verts + self._ipc_sphere_center
                        V_f = np.asfortranarray(np.vstack([pos_np, sphere_V]))
                        V_prev_f = np.asfortranarray(np.vstack([self._ipc_prev_V, sphere_V]))
                    else:
                        V_f = np.asfortranarray(pos_np)
                        V_prev_f = np.asfortranarray(self._ipc_prev_V)

                    dhat = self.ipc_dhat
                    nc = ipctk.NormalCollisions()
                    nc.build(cm, V_f, dhat=dhat)

                    if len(nc) > 0:
                        bp = ipctk.BarrierPotential(dhat=dhat)
                        kappa = self._ipc_kappa if self._ipc_kappa is not None else 1e6

                        tc = ipctk.TangentialCollisions()
                        tc.build(cm, V_f, nc, bp, kappa, mu)

                        if len(tc) > 0:
                            eps_v = 1e-3  # velocity mollifier
                            fp = ipctk.FrictionPotential(eps_v=eps_v)
                            friction_grad = fp.gradient(tc, cm, V_f, V_prev_f)
                            # Slice to shell-only and apply as velocity correction
                            friction_forces = friction_grad[:n_shell * 3].reshape(-1, 3)
                            # Scale friction forces by inverse mass for velocity update
                            inv_mass_np = model.particle_inv_mass.numpy()
                            for i in range(n_shell):
                                if inv_mass_np[i] > 0:
                                    vel_np[i] -= friction_forces[i] * inv_mass_np[i] * sub_dt

                            state_out.particle_qd = wp.array(
                                vel_np.astype(np.float32), dtype=wp.vec3, device=device
                            )
                except Exception:
                    pass  # Friction is best-effort; don't crash the sim

                self._ipc_prev_V = pos_np.copy()

        self.integrate_bodies(model, state_in, state_out, dt)

    def set_contact_sphere(self, center, radius, stiffness=1e4, damping=10.0):
        """Configure a simple sphere + ground contact for testing.

        Args:
            center: Sphere center as (x, y, z) tuple.
            radius: Sphere radius.
            stiffness: Contact penalty stiffness (keep low for implicit stability).
            damping: Contact damping coefficient.
        """
        self._contact_sphere_center = wp.vec3(float(center[0]), float(center[1]), float(center[2]))
        self._contact_sphere_radius = float(radius)
        self._contact_stiffness = float(stiffness)
        self._contact_damping = float(damping)

    def set_ipc_ground(self, z=0.0, normal=None, friction_coefficient=0.0):
        """Configure IPC barrier-based ground plane contact.

        Args:
            z: Ground plane height (default 0).
            normal: Ground plane normal as (x, y, z) tuple. Default (0, 0, 1).
            friction_coefficient: Coulomb friction coefficient (0 = frictionless).
        """
        if not self.use_ipc:
            raise RuntimeError("set_ipc_ground requires use_ipc=True")
        if normal is None:
            normal = (0.0, 0.0, 1.0)
        self._ipc_ground_z = float(z)
        n = np.array(normal, dtype=np.float64)
        n = n / np.linalg.norm(n)
        self._ipc_ground_normal = n
        # Compute a point on the plane: z * normal (for tilted planes)
        self._ipc_ground_origin = n * z
        self._ipc_friction_mu = float(friction_coefficient)

    def set_ipc_sphere(self, center, radius, friction_coefficient=0.0, subdivisions=3):
        """Configure IPC barrier-based sphere contact via unified CollisionMesh.

        The sphere is tessellated as an icosphere and combined with the shell
        mesh into a unified CollisionMesh. ipctk treats shell-vs-sphere as
        self-collision within the unified mesh. Sphere vertices are kinematic
        (not affected by contact forces).

        Can be called each step to update sphere position (animated spheres).

        Args:
            center: Sphere center as (x, y, z) tuple.
            radius: Sphere radius.
            friction_coefficient: Coulomb friction coefficient (0 = frictionless).
            subdivisions: Icosphere subdivision level (3 = ~162 verts, 4 = ~642 verts).
        """
        if not self.use_ipc:
            raise RuntimeError("set_ipc_sphere requires use_ipc=True")

        center = np.array(center, dtype=np.float64)
        radius = float(radius)

        # Only re-tessellate if radius changed or first call
        if (self._ipc_sphere_verts is None or
                self._ipc_sphere_radius != radius):
            verts, faces = self._tessellate_icosphere(subdivisions)
            self._ipc_sphere_verts = verts * radius  # unit sphere → scaled
            self._ipc_sphere_faces = faces

        self._ipc_sphere_center = center
        self._ipc_sphere_radius = radius
        self._ipc_sphere_friction = float(friction_coefficient)
        # Invalidate unified collision mesh (sphere position changed)
        self._ipc_unified_collision_mesh = None

    @staticmethod
    def _tessellate_icosphere(subdivisions=3):
        """Generate unit icosphere mesh via recursive subdivision.

        Returns:
            (vertices, faces): vertices as (n, 3) float64, faces as (t, 3) int32.
        """
        # Start from icosahedron
        phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
        verts = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ], dtype=np.float64)
        # Normalize to unit sphere
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ], dtype=np.int32)

        # Subdivide
        for _ in range(subdivisions):
            edge_midpoint = {}  # (min_idx, max_idx) → new vertex index
            new_faces = []
            verts_list = list(verts)

            for f in faces:
                mids = []
                for i in range(3):
                    e = tuple(sorted((f[i], f[(i + 1) % 3])))
                    if e not in edge_midpoint:
                        mid = (verts_list[e[0]] + verts_list[e[1]]) / 2.0
                        mid = mid / np.linalg.norm(mid)  # project to sphere
                        edge_midpoint[e] = len(verts_list)
                        verts_list.append(mid)
                    mids.append(edge_midpoint[e])

                v0, v1, v2 = f[0], f[1], f[2]
                m01, m12, m20 = mids[0], mids[1], mids[2]
                new_faces.extend([
                    [v0, m01, m20],
                    [v1, m12, m01],
                    [v2, m20, m12],
                    [m01, m12, m20],
                ])

            verts = np.array(verts_list, dtype=np.float64)
            faces = np.array(new_faces, dtype=np.int32)

        return verts, faces

    def _build_ipc_collision_mesh(self):
        """Build ipctk CollisionMesh from model triangle indices (cached).

        If a sphere is configured, builds a unified mesh (shell + sphere)
        for combined self-collision detection.
        """
        # Shell-only mesh (always needed)
        if self._ipc_collision_mesh is None:
            tri_np = self.model.tri_indices.numpy()  # (T, 3) int32
            n_verts = self.model.particle_count
            V_dummy = np.zeros((n_verts, 3), dtype=np.float64, order='F')
            E = ipctk.edges(tri_np)
            self._ipc_collision_mesh = ipctk.CollisionMesh(V_dummy, E, tri_np)
            self._ipc_tri_np = tri_np
            self._n_shell_verts = n_verts

        # If no sphere, return shell-only mesh
        if self._ipc_sphere_center is None:
            return self._ipc_collision_mesh, self._n_shell_verts

        # Build unified mesh (shell + sphere) — rebuilt when sphere moves
        if self._ipc_unified_collision_mesh is None:
            n_shell = self._n_shell_verts
            shell_tri = self._ipc_tri_np

            # Position sphere vertices at current center
            sphere_V = self._ipc_sphere_verts + self._ipc_sphere_center
            sphere_F = self._ipc_sphere_faces + n_shell  # offset indices

            n_total = n_shell + len(sphere_V)
            V_dummy = np.zeros((n_total, 3), dtype=np.float64, order='F')
            V_dummy[:n_shell] = 0.0  # shell positions filled at query time
            V_dummy[n_shell:] = sphere_V

            F_unified = np.vstack([shell_tri, sphere_F]).astype(np.int32)
            E_unified = ipctk.edges(F_unified)
            self._ipc_unified_collision_mesh = ipctk.CollisionMesh(
                V_dummy, E_unified, F_unified
            )

        return self._ipc_unified_collision_mesh, self._n_shell_verts

    def _compute_ipc_contact(self, V):
        """Compute IPC barrier gradient and diagonal hessian blocks for ground plane.

        Args:
            V: Vertex positions as (n, 3) float64 numpy array.

        Returns:
            (gradient, hess_diag): gradient is (n, 3) forces, hess_diag is (n, 3, 3) blocks.
        """
        n = V.shape[0]
        grad = np.zeros((n, 3), dtype=np.float64)
        hess_diag = np.zeros((n, 3, 3), dtype=np.float64)

        if self._ipc_ground_z is not None:
            V_f = np.asfortranarray(V)
            dhat = self.ipc_dhat
            bp = ipctk.BarrierPotential(dhat=dhat)

            # Construct plane collisions
            origin = np.asfortranarray(self._ipc_ground_origin.reshape(1, 3))
            normal = np.asfortranarray(self._ipc_ground_normal.reshape(1, 3))
            plane_collisions = ipctk.construct_point_plane_collisions(
                V_f, origin, normal, dhat=dhat
            )

            # Auto-estimate barrier stiffness on first call
            if self._ipc_kappa is None and len(plane_collisions) > 0:
                # Adaptive barrier stiffness (#50) via ipctk
                try:
                    bbox_diag = ipctk.world_bbox_diagonal_length(V_f)
                    avg_mass = 1.0 / max(np.mean(self.model.particle_inv_mass.numpy()), 1e-10)
                    # Compute barrier gradient for stiffness estimation
                    _barrier = ipctk.BarrierPotential(dhat=dhat)
                    _nc_tmp = ipctk.NormalCollisions()
                    _cm, _ = self._build_ipc_collision_mesh()
                    _nc_tmp.build(_cm, V_f, dhat=dhat)
                    if len(_nc_tmp) > 0:
                        _grad_barrier = _barrier.gradient(_nc_tmp, _cm, V_f)
                        _grad_energy = np.zeros_like(_grad_barrier)  # approximate
                        self._ipc_kappa, self._ipc_max_kappa = ipctk.initial_barrier_stiffness(
                            bbox_diag, ipctk.barrier, dhat, avg_mass,
                            _grad_energy.reshape(-1, 1), _grad_barrier.reshape(-1, 1)
                        )
                    else:
                        self._ipc_kappa = avg_mass * 1e4
                except Exception:
                    avg_mass = 1.0 / max(np.mean(self.model.particle_inv_mass.numpy()), 1e-10)
                    self._ipc_kappa = avg_mass * 1e4  # fallback heuristic

            kappa = self._ipc_kappa if self._ipc_kappa is not None else 1e6

            for pc in plane_collisions:
                vid = pc.vertex_id
                x = np.asfortranarray(V[vid].reshape(3, 1))
                g = bp.gradient(pc, x).flatten()
                h = bp.hessian(pc, x,
                               project_hessian_to_psd=ipctk.PSDProjectionMethod.CLAMP)
                grad[vid] += kappa * g
                hess_diag[vid] += kappa * h

        return grad, hess_diag

    def _compute_ipc_self_collision(self, V):
        """Compute IPC self-collision barrier gradient and sparse hessian.

        If a sphere is configured, uses the unified mesh (shell + sphere)
        and slices out shell-only forces/hessian.

        Args:
            V: Shell vertex positions (n_shell, 3) float64.

        Returns:
            (gradient, hessian_csr): gradient (n_shell*3,), hessian as scipy CSR or None.
        """
        cm, n_shell = self._build_ipc_collision_mesh()
        dhat = self.ipc_dhat

        # Build unified vertex array if sphere is present
        if self._ipc_sphere_center is not None:
            sphere_V = self._ipc_sphere_verts + self._ipc_sphere_center
            V_unified = np.vstack([V, sphere_V])
        else:
            V_unified = V

        V_f = np.asfortranarray(V_unified)

        nc = ipctk.NormalCollisions()
        nc.build(cm, V_f, dhat=dhat)

        if len(nc) == 0:
            return np.zeros(n_shell * 3, dtype=np.float64), None

        bp = ipctk.BarrierPotential(dhat=dhat)
        kappa = self._ipc_kappa if self._ipc_kappa is not None else 1e6

        grad_full = kappa * bp.gradient(nc, cm, V_f)

        # Try full PSD-projected hessian; fall back to None if PSD projection fails
        try:
            hess_full = kappa * bp.hessian(nc, cm, V_f,
                                            project_hessian_to_psd=ipctk.PSDProjectionMethod.CLAMP)
        except RuntimeError:
            # PSD projection can fail on degenerate configurations
            hess_full = None

        # Slice to shell-only vertices
        n3 = n_shell * 3
        grad = grad_full[:n3]

        # Slice hessian to shell-only block (top-left n3 x n3)
        if hess_full is not None:
            hess = hess_full[:n3, :n3]
        else:
            hess = None

        return grad, hess

    def _ipc_ccd_step_size(self, V_current, V_candidate):
        """Compute maximum collision-free step size via CCD.

        Uses unified mesh (shell + sphere) when sphere is configured.

        Args:
            V_current: Current shell positions (n_shell, 3) float64.
            V_candidate: Candidate shell positions (n_shell, 3) float64.

        Returns:
            float: Maximum safe step fraction in [0, 1].
        """
        alpha = 1.0

        # Ground plane CCD
        if self._ipc_ground_z is not None:
            origin = np.asfortranarray(self._ipc_ground_origin.reshape(1, 3))
            normal = np.asfortranarray(self._ipc_ground_normal.reshape(1, 3))
            alpha_ground = ipctk.compute_point_plane_collision_free_stepsize(
                np.asfortranarray(V_current),
                np.asfortranarray(V_candidate),
                origin, normal
            )
            alpha = min(alpha, alpha_ground)

        # Self-collision CCD (with unified mesh if sphere present)
        cm, n_shell = self._build_ipc_collision_mesh()

        if self._ipc_sphere_center is not None:
            # Unified mesh: shell vertices move, sphere vertices stay fixed
            sphere_V = self._ipc_sphere_verts + self._ipc_sphere_center
            V_cur_unified = np.vstack([V_current, sphere_V])
            V_cand_unified = np.vstack([V_candidate, sphere_V])  # sphere doesn't move within substep
        else:
            V_cur_unified = V_current
            V_cand_unified = V_candidate

        alpha_self = ipctk.compute_collision_free_stepsize(
            cm,
            np.asfortranarray(V_cur_unified),
            np.asfortranarray(V_cand_unified),
        )
        alpha = min(alpha, alpha_self)

        # Safety margin: use 0.8 * alpha to stay within barrier activation zone
        return 0.8 * alpha
