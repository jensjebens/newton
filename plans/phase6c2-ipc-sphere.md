# Phase 6c.2: IPC Sphere Contact via Unified CollisionMesh

> Replace penalty-based `set_contact_sphere()` with ipctk barrier contact using a unified mesh approach.

## Motivation

The current FEM shell solver uses two contact systems:
- **IPC barriers** (ipctk) for ground plane + self-collision — zero penetration, CCD, friction
- **Penalty forces** (custom Warp kernel) for sphere contact — tunable stiffness, allows penetration

The ball-in-bag demo runs on penalty sphere contact. This means:
1. Sphere penetration is possible at high velocities or low stiffness
2. No CCD for sphere tunneling
3. No proper friction between sheet and sphere
4. Two competing contact models in the same solver = complexity + parameter tuning

**Goal:** Unify all contact through ipctk by treating the sphere as part of the collision mesh.

## Approach: Unified CollisionMesh

Verified working in the previous session with live code:

```python
# 1. Tessellate sphere into triangle mesh
sphere_V, sphere_F = tessellate_icosphere(center, radius, subdivisions=3)
# ~162 vertices, ~320 triangles — sufficient for smooth contact

# 2. Combine bag + sphere into one mesh
V_unified = np.vstack([bag_V, sphere_V])           # (n_bag + n_sphere, 3)
F_unified = np.vstack([bag_F, sphere_F + n_bag])    # offset sphere face indices

# 3. Build CollisionMesh — ipctk handles bag-vs-sphere as "self-collision"
collision_mesh = ipctk.CollisionMesh(V_unified, edges, F_unified)

# 4. After computing barrier gradient over all vertices:
grad_bag = grad[:n_bag * 3]        # only bag vertices get forces
# sphere vertices are kinematic — zero inverse mass, no force application

# 5. Hessian: extract bag-only block from sparse matrix
H_bag = hess[:n_bag*3, :n_bag*3]   # top-left block
```

**Why this works:**
- ipctk's vertex-triangle and edge-edge collision detection naturally finds bag-sphere contacts
- Sphere vertices are kinematic (updated from animation, not from forces)
- Gradient slicing gives bag-only forces — verified with actual ipctk output
- Hessian block extraction gives bag-only stiffness matrix — verified with scipy slicing
- CCD works on the unified mesh — prevents both self-intersection AND sphere tunneling

## Acceptance Criteria

1. **`set_ipc_sphere(center, radius)` API** — configures sphere as IPC obstacle
2. **Sphere contact via ipctk barriers** — zero penetration between sheet and sphere
3. **Sphere is kinematic** — animated position updated each substep, no forces applied to sphere
4. **CCD includes sphere** — `compute_collision_free_stepsize` on unified mesh
5. **Friction on sphere surface** — ipctk TangentialCollisions between bag and sphere
6. **Backward compatible** — `set_contact_sphere()` still works as fallback penalty
7. **Existing 16 tests still pass**
8. **4 new tests** for IPC sphere behavior
9. **Ball-in-bag demo** re-rendered with IPC sphere (comparison GIF)

## Implementation Steps

### Step 1: Icosphere tessellation utility
```python
def tessellate_icosphere(center, radius, subdivisions=3):
    """Generate triangle mesh for a sphere via icosahedron subdivision.
    Returns (vertices, faces) as numpy arrays.
    ~162 verts / ~320 faces at subdivisions=3.
    """
```
Pure numpy, no external dependencies. Start from icosahedron, subdivide + project.

### Step 2: `set_ipc_sphere()` method on SolverFEMShell
- Stores sphere center, radius, and tessellation
- Sphere positions updated each substep (for animated spheres)
- Replaces penalty path when `use_ipc=True`

### Step 3: Unified CollisionMesh builder
Extend `_build_ipc_collision_mesh()`:
- If sphere configured, append sphere vertices/faces to bag mesh
- Track `n_bag` boundary for gradient/hessian slicing
- Cache tessellation, rebuild unified mesh when sphere moves

### Step 4: Gradient and Hessian slicing
Modify `_compute_ipc_self_collision()`:
- Run on unified mesh (bag + sphere)
- Slice gradient: `grad[:n_bag*3]` for bag-only forces
- Slice hessian: `hess[:n_bag*3, :n_bag*3]` for bag-only stiffness
- Update sphere positions from animation (not from gradient)

### Step 5: CCD on unified mesh
Modify `_ipc_ccd_step_size()`:
- Build candidate positions for unified mesh (bag candidates + sphere current positions)
- `compute_collision_free_stepsize` on unified mesh prevents bag-sphere tunneling

### Step 6: Friction via TangentialCollisions
After NormalCollisions on unified mesh:
```python
tc = ipctk.TangentialCollisions()
np_ = ipctk.BarrierPotential(dhat=dhat)  # normal potential
tc.build(cm, V_unified, nc, np_, kappa, mu=friction_coefficient)

eps_v = 1e-3  # velocity mollifier
fp = ipctk.FrictionPotential(eps_v=eps_v)
friction_grad = fp.gradient(tc, cm, V_unified, V_prev)
# Slice bag-only friction forces: friction_grad[:n_bag*3]
```

### Step 7: Adaptive barrier stiffness
Wire in `ipctk.initial_barrier_stiffness()`:
```python
barrier = ipctk.barrier  # or the module-level barrier function
bbox_diag = ipctk.world_bbox_diagonal_length(V)
avg_mass = total_mass / n_verts
grad_energy = elastic_gradient  # from FEM
grad_barrier = bp.gradient(nc, cm, V)  # bp = BarrierPotential
kappa, max_kappa = ipctk.initial_barrier_stiffness(
    bbox_diag, ipctk.barrier, dhat, avg_mass, grad_energy, grad_barrier
)
```

## Test Plan

### Test 17: IPC sphere contact — zero penetration
Sheet drapes onto sphere with `use_ipc=True` + `set_ipc_sphere()`. After settling, minimum distance from any bag vertex to sphere surface > 0. Compare with penalty: penalty allows ~1mm penetration, IPC should be exact.

### Test 18: Animated sphere — kinematic update
Sphere moves downward through a horizontal sheet (like ball-in-bag). Sphere position comes from animation, bag deforms around it. Verify sphere follows prescribed trajectory exactly (not affected by contact forces).

### Test 19: CCD sphere tunneling prevention
Sphere with high velocity (-10 m/s) aimed at sheet. Without CCD it tunnels through. With unified CCD, all bag vertices remain on the correct side of the sphere.

### Test 20: Friction on sphere surface
Sheet draped on top of sphere with friction μ=0.5. Tilt gravity to 30° — sheet should slide slowly (dynamic friction), not fly off. Compare center-of-mass displacement with and without friction.

## Key Parameters

- `sphere_subdivisions`: 3 (default) — ~162 verts, good enough for smooth contact
- `dhat`: same as ground (0.01m), may need tuning for sphere curvature
- `friction_coefficient`: per-surface or global, default 0.3

## Performance Notes

- Icosphere tessellation: one-time cost, ~0.1ms
- Unified mesh: ~560 verts (400 bag + 162 sphere) for 20×20 grid — still fast for ipctk
- Main overhead: NormalCollisions.build on larger mesh — profile and report

## Risks

| Risk | Mitigation |
|------|------------|
| Hessian slicing breaks sparsity pattern | Use scipy CSR slicing (efficient for row selection) |
| Sphere tessellation too coarse → contact gaps | Increase subdivisions to 4 (~642 verts) if needed |
| Unified CCD too conservative (tiny steps) | Tune safety margin (currently 0.8) |
| Friction on curved surface unstable | Start with frictionless sphere, add friction incrementally |

## Dependencies

- ipctk >= 1.5.0 ✅
- scipy (CSR slicing) ✅
- No new deps
