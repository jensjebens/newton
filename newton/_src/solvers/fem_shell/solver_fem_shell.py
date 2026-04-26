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
import warp as wp
import warp.sparse as wps
from warp.optim.linear import cg, preconditioner

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

            # 1c. Simple contact forces (ground + sphere)
            # TODO Phase C: replace with proper IPC barriers
            if hasattr(self, '_contact_sphere_center'):
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
            n_total = n_mem + n_bend

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
            if hasattr(self, '_contact_sphere_center'):
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
