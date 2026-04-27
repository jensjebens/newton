# Phase 6c: IPC Contact Integration

> Replace custom penalty contacts with ipctk barrier potentials for intersection-free contact.

## Motivation

Current contact handling uses custom penalty forces (cubic stiffness + projection). This works for simple ground+sphere cases but:
1. No self-collision — sheet can fold through itself
2. No friction — sheet slides freely on contact surfaces  
3. Penalty stiffness is a tuning parameter — too low = penetration, too high = instability
4. No CCD (continuous collision detection) — fast motion can tunnel through obstacles

IPC (Incremental Potential Contact) guarantees zero interpenetration via barrier potentials with CCD-limited step sizes. `ipctk` v1.5.0 is installed and verified working.

## Acceptance Criteria

1. **IPC barrier replaces penalty contact** for ground plane collisions — zero penetration guaranteed
2. **Self-collision** — two sheets or a folding sheet cannot pass through itself
3. **CCD step size limiting** — `compute_collision_free_stepsize()` prevents tunneling
4. **Friction** (basic) — sheet doesn't slide freely on contact surfaces
5. **Backward compatible** — existing 12 tests still pass, set_contact_sphere() still works as fallback
6. **New tests** — at least 4 new tests covering IPC-specific behavior
7. **Demo** — same cloth-on-sphere drape with IPC contact, generates GIF + USD for comparison

## Approach

### ipctk Integration Pattern

ipctk runs on CPU with numpy arrays. The integration loop per substep:

```python
# 1. Transfer positions from GPU → CPU
V = state.particle_q.numpy()  # (n, 3) float64

# 2. Build/update collision mesh
collision_mesh = ipctk.CollisionMesh(V, faces)  # faces = triangle indices

# 3. Detect collisions (broad + narrow phase)
collisions = ipctk.NormalCollisions()
collisions.build(collision_mesh, V, dhat)  # dhat = barrier activation distance

# 4. Compute barrier potential gradient (= contact forces)
barrier = ipctk.BarrierPotential(dhat)
grad = barrier.gradient(collisions, collision_mesh, V)  # (n*3,) array

# 5. Compute barrier Hessian (for implicit solve)
hess = barrier.hessian(collisions, collision_mesh, V)  # sparse CSR matrix

# 6. Transfer forces back to GPU, add to RHS
# 7. CCD step size: after velocity solve, limit step to collision-free
V_new = V + dt * vel
max_step = ipctk.compute_collision_free_stepsize(
    collision_mesh, V, V_new
)
```

### Ground Plane via ipctk

Use `ipctk.construct_point_plane_collisions()` and `ipctk.compute_point_plane_collision_free_stepsize()` for ground plane instead of custom penalty kernel. This gives us barrier-based ground contact with CCD.

### Self-Collision

ipctk handles vertex-triangle and edge-edge collisions natively. For a single sheet:
- Vertex-triangle: particle vs non-adjacent triangles
- Edge-edge: non-adjacent edge pairs

The `CollisionMesh` constructor with `can_collide` matrix controls adjacency filtering.

### Integration into SolverFEMShell.step()

1. After elastic force computation, transfer positions to CPU
2. Run ipctk collision detection + barrier gradient/hessian
3. Add barrier gradient to RHS forces (GPU)
4. Add barrier hessian to stiffness matrix (convert CSR → BSR triplets)
5. After velocity solve, CCD line search to limit position update
6. Remove custom penalty kernel when IPC is active

### Performance Note

CPU↔GPU transfers for a 24×24 grid (~625 vertices): negligible.
ipctk collision detection: fast for <5K vertices.
If perf becomes an issue, defer to Phase D optimization.

## Key Parameters

- `dhat` (barrier activation distance): start with 0.01 (1cm), tune per scene
- `barrier_stiffness`: use `ipctk.initial_barrier_stiffness()` for automatic estimation
- `friction_coefficient`: 0.3 (default), per-material override
- Adaptive stiffness: `ipctk.update_barrier_stiffness()` each step

## Test Plan

### Test 7: IPC ground contact — zero penetration
Sheet drops onto ground plane. After settling, ALL particles must be at z >= 0 (within floating point tolerance). No custom penalty — pure IPC barrier.

### Test 8: Self-collision — folding sheet
Sheet pinned at two ends, bent in half (or gravity + obstacle forces it to fold). The two halves must NOT interpenetrate. Measure minimum distance between non-adjacent faces > 0.

### Test 9: CCD step size — fast impact
Sheet with high initial velocity (-20 m/s) aimed at ground. Without CCD it would tunnel through. With CCD, `compute_collision_free_stepsize()` limits the step. All particles must remain above ground.

### Test 10: Friction — sheet on slope
Sheet on a 30° slope with friction coefficient 0.5. Static friction should prevent sliding (tan(30°) ≈ 0.577 > μ... wait, that means it should slide). Use 20° slope where tan(20°) ≈ 0.364 < 0.5 — sheet should NOT slide off.

## Dependencies

- `ipctk >= 1.5.0` ✅ installed
- `scipy` for sparse matrix conversion (CSR → triplets) — should already be available
- No new GPU dependencies

## Risks

| Risk | Mitigation |
|------|------------|
| CPU↔GPU transfer overhead | Profile first; only ~625 verts for test scenes |
| ipctk barrier too stiff for PCG | Use `initial_barrier_stiffness()` adaptive, increase CG iterations |
| Sparse format conversion (CSR→BSR) slow | Cache sparsity pattern, only update values |
| Self-collision false positives on thin sheets | Tune `dhat` relative to thickness |
