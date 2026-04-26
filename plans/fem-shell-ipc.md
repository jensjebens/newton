# FEM Thin Shell Solver + IPC Contact for Newton

> Standalone project — GPU-accelerated FEM shell solver for stiff thin materials in Newton.
> Repo: `jensjebens/newton` branch `feature/fem-shell-solver`

## Baseline Results (VBD Solver — "Before")

Ran cardboard simulation with VBD to establish baseline for comparison.

### Setup
- 16×16 grid, 5cm cells → 0.80m × 0.80m sheet, 289 particles, 512 triangles
- Left edge fixed (cantilever), gravity, ground plane
- 120 frames @ 60fps, GPU (L40)

### Test 1: Normal stiffness (tri_ke=1e5, edge_ke=1e4, 20 iters, 4 substeps)
| Metric | Result | Expected (cardboard) |
|--------|--------|--------------------|
| Stretch ratio | 1.36x | ~1.0 (inextensible) |
| Max sag | 0.16-1.0m oscillating | ~0.05m steady |
| Settled? | No (still oscillating at 2s) | Should settle <1s |

### Test 2: Extreme stiffness (tri_ke=1e7, edge_ke=1e6, 50 iters, 16 substeps)
| Metric | Result | Expected |
|--------|--------|----------|
| Stretch ratio | 0.86x (compressed!) | ~1.0 |
| Max sag | 0.20-0.74m oscillating | ~0.05m |
| Settled? | No | Should settle |
| Compute cost | 4x baseline | — |

### Conclusion
VBD cannot properly simulate stiff thin shells like cardboard. Even with 100x stiffness params, 2.5x iterations, and 4x substeps, the sheet:
- Still stretches/compresses significantly
- Never reaches steady state (oscillates indefinitely)
- Bending behavior is cloth-like, not plate-like

This validates the need for a proper FEM shell formulation with implicit time integration.

## Motivation

Newton's current soft body solvers (VBD, XPBD, Style3D) are designed primarily for cloth and volumetric deformables. For stiff thin-shell materials like **cardboard, sheet metal, plastic panels** — materials with high stretch resistance and significant bending stiffness — we need a proper FEM thin shell formulation.

Additionally, folding/stacking scenarios (e.g. cardboard boxes) require robust self-collision handling that prevents interpenetration. IPC (Incremental Potential Contact) is the state-of-the-art solution for this.

**Goal:** Add a GPU-accelerated FEM thin shell solver to Newton using Warp FEM for the shell mechanics and IPC Toolkit for contact handling.

### Why not existing Newton solvers? (Verified April 26)

We systematically tested **every available solver** before concluding a new one is needed:

| Solver | Test | Result |
|--------|------|--------|
| **VBD** (normal) | Cardboard params, 20 iters, 4 substeps | 36% stretch, oscillates forever |
| **VBD** (extreme) | 100 iters, 32 substeps (max compute) | ✅ 0% stretch, but sags 0.85m — membrane works, **bending doesn't** |
| **VBD** (high bending) | edge_ke=1e7 | 💥 **Diverges** — falls through ground |
| **VBD** (thin tets) | 2mm tet slab (soft_grid, dim_z=1) | 💥 **NaN** after ~45 frames |
| **XPBD** | 50 iters, 16 substeps | 💥 **6000x lateral stretch** explosion |
| **Style3D** | — | 🔴 CUDA crash (#2586, #2587) |
| **ImplicitMPM** | — | Volumetric/granular only, can't represent thin geometry |
| **Kamino** | — | Rigid bodies only |

**Key finding:** VBD's StVK membrane energy on triangles is excellent — with enough iterations it achieves perfect inextensibility. The gap is specifically **bending**: VBD uses dihedral angle hinge constraints (edge_ke), which:
- Work great for soft cloth (small deformations around rest angle)
- **Diverge** when stiffness is high (Gauss-Seidel local solve can't converge)
- Have no middle ground between "too soft" and "unstable"

This is not a parameter tuning issue — it's a fundamental limitation of the bending model.

### Why Warp FEM + IPC?

- **Warp FEM** is already Newton's compute backend — same CUDA runtime, same arrays, same device. `Trimesh3D` provides triangle elements embedded in 3D space (exactly what shell FEM needs). GPU-native, differentiable.
- **IPC Toolkit** (`ipctk` v1.5.0) provides intersection-free, inversion-free contact with friction. CPU-based but lightweight for the contact resolution step. Python API via `pip install ipctk`.

## Acceptance Criteria

1. **Solver class:** `SolverFEMShell` inheriting from `SolverBase`, following the same `step(state_0, state_1, control, contacts, dt)` API pattern as VBD/XPBD
2. **Shell FEM mechanics:** Kirchhoff-Love or Discrete Shells formulation on `Trimesh3D` geometry
   - Membrane energy (stretch resistance) — high stiffness for cardboard
   - Bending energy (curvature resistance) — dihedral angle based
   - GPU-accelerated via Warp kernels
3. **IPC contact:** Self-collision and object-collision using IPC Toolkit barrier potentials
   - Collision mesh built from Newton's particle positions
   - Barrier gradient/hessian fed back as forces into the FEM solve
   - Collision-free CCD step size limiting
4. **Newton integration:** Works with existing Newton `ModelBuilder` — users call `builder.add_cloth_grid()` or `builder.add_cloth_mesh()` and select `solver_type="fem_shell"`
5. **USD/UsdView:** Simulated mesh plays back in UsdView via our existing HdExec pipeline
6. **Tests:**
   - Flat sheet under gravity (drape test) — verify bending energy works
   - Sheet on ramp (friction test) — verify IPC friction
   - Two sheets collision (self-collision test) — verify IPC prevents interpenetration
   - Stiff material (cardboard) — verify high Young's modulus doesn't explode
   - Compare against VBD on same scene for visual plausibility
7. **Performance:** Real-time-ish (>5 fps) for a 32×32 quad sheet on L40

## Approach

### Architecture

```
Newton ModelBuilder
    ↓
SolverFEMShell
    ├── Warp FEM (Trimesh3D)
    │   ├── Membrane energy kernel (GPU)
    │   ├── Bending energy kernel (GPU)  
    │   └── Newton-Raphson implicit solve (GPU)
    ├── IPC Toolkit (CPU)
    │   ├── Collision detection (broad + narrow phase CCD)
    │   ├── Barrier potential + gradient + hessian
    │   └── Friction potential
    └── Time integration
        ├── Implicit Euler (for stiffness stability)
        └── Line search with IPC step size limit
```

### Phase A: Membrane-only FEM shell (no bending, no IPC)

1. Create `newton/_src/solvers/fem_shell/` directory
2. Implement membrane energy using Warp FEM `Trimesh3D`:
   - St. Venant-Kirchhoff or Neo-Hookean constitutive model
   - Compute deformation gradient F from reference → current config
   - Strain energy + forces + stiffness matrix via Warp kernels
3. Implicit Euler time integration (GPU sparse solve via `warp.sparse`)
4. Test: flat sheet stretching under gravity → should resist stretching

### Phase B: Bending energy (curvature-based, NOT dihedral angle)

**IMPORTANT:** VBD already uses dihedral angle bending (edge_ke) and we proved it diverges at high stiffness. We must NOT repeat the same approach. Use curvature-based discrete shells instead:

1. Implement **Discrete Kirchhoff Triangle (DKT)** bending elements:
   - Curvature tensor computed from vertex positions + edge normals
   - Bending energy = ½ h³/12 · (κ - κ₀)ᵀ D (κ - κ₀) integrated over element area
   - Where D is the bending constitutive matrix (from E, ν)
   - This couples bending stiffness to material thickness cubically (h³) — physically correct
2. Alternative: **Hinge-based Discrete Shells** (Grinspun 2003) but with **global implicit solve** instead of VBD's local Gauss-Seidel — the divergence issue is the local solve, not the energy itself
3. Warp kernel for element-level bending force + stiffness matrix
4. Test: cantilever beam (one side fixed, gravity) → compare deflection against Euler-Bernoulli analytical solution (δ = qL⁴/8EI where I = h³/12)

### Phase C: IPC contact integration

1. Build `ipctk.CollisionMesh` from Newton particle positions each step
2. Compute IPC barrier potential, gradient, and hessian
3. Transfer IPC forces from CPU → GPU (small overhead, contact pairs are sparse)
4. Add IPC CCD step size limiting to the line search
5. Test: two sheets falling onto each other → no interpenetration

### Phase D: Newton solver integration + USD

1. Register `SolverFEMShell` in Newton's solver registry
2. Wire into `ModelBuilder.add_cloth_grid()` / `add_cloth_mesh()` with `solver_type="fem_shell"`
3. Material parameters: Young's modulus E, Poisson's ratio ν, thickness h, density ρ
   - Cardboard preset: E ≈ 1-5 GPa, ν ≈ 0.3, h ≈ 1-3mm, ρ ≈ 600-800 kg/m³
4. UsdView plugin: reuse existing Newton→HdExec→Storm pipeline
5. Demo: cardboard sheet falling onto table, folding, stacking

## Material Properties Reference

| Material | Young's Modulus (E) | Poisson's Ratio (ν) | Thickness (h) | Density (ρ) |
|----------|-------------------|---------------------|---------------|-------------|
| Paper | 1-10 GPa | 0.1-0.3 | 0.1 mm | 700-1200 kg/m³ |
| Cardboard | 1-5 GPa | 0.2-0.4 | 1-3 mm | 150-800 kg/m³ |
| Sheet metal (steel) | 200 GPa | 0.3 | 0.5-3 mm | 7800 kg/m³ |
| Plastic (ABS) | 2-3 GPa | 0.35 | 1-5 mm | 1050 kg/m³ |

## Key References

- **Discrete Shells** — Grinspun et al., SCA 2003 — dihedral angle bending for triangle meshes
- **IPC** — Li et al., SIGGRAPH 2020 — intersection-free contact
- **Codim-IPC** — Li et al., SIGGRAPH 2021 — codimensional (thin shell) IPC
- **Warp FEM** — NVIDIA, 2022+ — GPU FEM framework with Trimesh3D
- **Stable Neo-Hookean** — Smith et al., 2018 — robust hyperelastic membrane energy

## Open Questions (Researched)

1. ✅ **Warp FEM Trimesh3D in 3D:** CONFIRMED — `Trimesh3D` has `dimension=3`, natively handles 3D-embedded triangles. Function spaces and gradient computation work in 3D. Side count gives us edges for bending energy. No extensions needed.
2. **IPC CPU↔GPU transfer cost:** The IPC Toolkit runs on CPU. For a 32×32 mesh (~2K triangles), the transfer should be negligible. For larger meshes (100K+ triangles), we may need to port IPC barrier computation to Warp kernels. → **Defer profiling to Phase 6c.**
3. ✅ **Sparse solver on GPU:** CONFIRMED — `warp.sparse` provides `BsrMatrix` (Block Sparse Row) with `bsr_mv` (matrix-vector), `bsr_mm` (matrix-matrix), `bsr_from_triplets`, `bsr_set_from_triplets`, `bsr_axpy`, `bsr_get_diag`. This is sufficient for PCG solver with block-diagonal preconditioning on GPU.
4. **Integration with Newton's collision system:** Newton's `model.collide()` handles rigid body contacts. IPC will run in parallel for deformable self-collision. Ground plane can be added as a static IPC obstacle. → **Design in Phase 6c.**
5. **Differentiability:** Both Warp and IPC are differentiable. Not a goal for Phase 6 (POC). → **Defer to Phase 7.**
6. ✅ **IPC Toolkit availability:** CONFIRMED — `ipctk` v1.5.0 installed and tested. Collision detection, barrier potential, gradient, hessian all work from Python.

## Dependencies

- `warp-lang >= 1.12.0` (already installed)
- `ipctk >= 1.5.0` (installed: `pip install ipctk`)
- `newton >= 1.1.0` (installed)
- No new C++ dependencies

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Warp FEM Trimesh3D doesn't support 3D-embedded shell elements | Medium | High | Fall back to custom Warp kernels for membrane/bending (bypass warp.fem) |
| IPC CPU transfer becomes bottleneck at scale | Low (for POC) | Medium | Profile first; port barrier kernel to Warp if needed |
| Implicit Euler GPU solve too slow | Low | Medium | Use PCG with Jacobi preconditioner; Warp has good sparse support |
| Material stiffness causes conditioning issues | Medium | Medium | Use adaptive time stepping or stiffness damping |
| Curvature-based bending diverges like dihedral | Low | High | DKT + global implicit solve is fundamentally different from VBD's local Gauss-Seidel; well-studied in FEM literature |

## Design Decision: New Solver vs Extending VBD

Considered extending VBD with better bending, but rejected because:
1. VBD's Gauss-Seidel iteration is the root cause of bending divergence, not just the energy model
2. A global implicit solve (assemble stiffness matrix, PCG) handles stiff bending naturally
3. VBD's graph coloring / parallel vertex solve doesn't map well to element-assembled FEM
4. Separate solver allows physical material params (E, ν, h, ρ) without breaking VBD's tuning-based API
5. VBD remains the right choice for soft cloth — this solver targets a different regime
