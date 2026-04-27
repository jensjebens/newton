# Plan: GPU FEA Solver (Structural Analysis)

> Branch: `plan/gpu-fea-solver` on `jensjebens/newton`
> Status: Plan only — no implementation yet
> Prerequisites: SolverFEMShell (branch `feature/fem-shell-solver`)

## Motivation

The FEM shell solver built for Newton already has the core infrastructure for structural finite element analysis:
- Global stiffness matrix assembly (BSR format)
- PCG linear solver (warp.optim.linear.cg)
- Physical material parameters (E, ν, h)
- StVK constitutive model (nonlinear elasticity)
- GPU acceleration via Warp

Extending this to a general-purpose FEA tool (à la simplified Ansys) requires adding static solve, boundary conditions, load cases, and stress output.

## Scope

A GPU-accelerated structural FEA solver for:
- **Shell elements** (existing) — thin plates, panels, sheet metal
- **Solid elements** (new) — 3D tetrahedral, based on VBD's Neo-Hookean tet kernels
- **Static + dynamic** analysis
- **Linear + nonlinear** material models

NOT in scope (future work):
- Thermal / multi-physics
- Fluid-structure interaction
- Adaptive meshing
- Higher-order elements (quadratic+)
- Buckling / stability analysis

## Architecture

```
FEASolver
├── Elements
│   ├── ShellTriangle (existing — StVK membrane + dihedral bending)
│   ├── SolidTetrahedron (new — Neo-Hookean from VBD)
│   └── (future: Beam, Quad, Hex)
├── Materials
│   ├── LinearElastic (new — small strain, Hooke's law)
│   ├── StVK (existing — large strain)
│   ├── NeoHookean (existing in VBD — port to global solve)
│   └── ElastoPlastic (existing — yield + permanent strain)
├── Boundary Conditions
│   ├── FixedDisplacement (existing partial — fix_left)
│   ├── PrescribedDisplacement (new)
│   ├── Symmetry (new)
│   └── Springs/Elastic Support (new)
├── Loads
│   ├── Gravity (existing)
│   ├── PointForce (new)
│   ├── Pressure (new — normal to surface)
│   ├── DistributedLoad (new — per-element)
│   └── ThermalLoad (future)
├── Solver
│   ├── StaticEquilibrium (new — Kx = f, no inertia)
│   ├── ImplicitDynamic (existing — (M + dt²K)dv = dt·f)
│   ├── ModalAnalysis (new — eigenvalue solve)
│   └── NonlinearNewtonRaphson (partially existing)
└── PostProcessing
    ├── Displacement field
    ├── Stress tensor (Cauchy / PK2)
    ├── Strain tensor (Green-Lagrange / engineering)
    ├── Von Mises stress
    ├── Principal stresses / directions
    ├── Reaction forces at BCs
    └── USD export (color-mapped stress visualization)
```

## Phases

### Phase 1: Static Shell FEA (1-2 days)
**Goal:** Solve Kx = f for shell elements with BCs and loads.

1. **Static solve mode:** Remove inertia term from system, solve `K·u = f` directly
   - Reuse existing stiffness assembly
   - Add `solve_static(model, bcs, loads) → displacements`
2. **General boundary conditions:**
   - `FixedBC(node_ids)` — zero displacement
   - `PrescribedBC(node_ids, displacement)` — known displacement
   - Implementation: zero rows/cols in K for fixed DOFs, modify RHS
3. **Load cases:**
   - `PointLoad(node_id, force_vector)`
   - `GravityLoad(direction, magnitude)`
   - `PressureLoad(face_ids, pressure)` — requires face normals
4. **Stress output:**
   - Compute Green-Lagrange strain E from deformation gradient F
   - Second Piola-Kirchhoff stress S = 2μE + λtr(E)I (already computed internally)
   - Von Mises: σ_vm = √(3/2 · S_dev : S_dev)
   - Output per-element or interpolated to vertices

**Test:** Cantilever beam under end load — compare deflection with analytical Euler-Bernoulli solution.

### Phase 2: Solid (Tet) Elements (1-2 days)
**Goal:** Add 3D tetrahedral elements for volumetric FEA.

1. **Neo-Hookean tet element:** Port from VBD's `evaluate_volumetric_neo_hookean_force_and_hessian`
2. **Stiffness assembly:** 4×4 vertex blocks per tet (12×12 element stiffness → 16 blocks)
3. **Mesh input:** `add_soft_grid` / `add_soft_mesh` from Newton ModelBuilder
4. **Mixed shell+solid:** Support both element types in one solve

**Test:** 3D block under compression — compare with analytical solution.

### Phase 3: USD Visualization (1 day)
**Goal:** Export FEA results as USD with color-mapped stress.

1. **Stress-to-color mapping:** Von Mises → heatmap (blue=low, red=high)
2. **Deformed mesh export:** Scale deformation for visualization (amplification factor)
3. **UsdGeom.Mesh with primvars:** Per-vertex color from interpolated stress
4. **UsdView rendering:** Storm renders the color-mapped result
5. **Animation:** If dynamic, export time-sampled stress fields

### Phase 4: Engineering Workflows (2-3 days)
**Goal:** Make it usable for actual engineering analysis.

1. **Multi-load-case solve:** Solve multiple RHS simultaneously (efficient with same K)
2. **Factor of safety:** Compare stress field against material yield strength
3. **Reaction forces:** Compute forces at boundary condition nodes
4. **Report generation:** Markdown/HTML report with stress plots, max values, safety factors
5. **Modal analysis:** Eigenvalue solve for natural frequencies (warp doesn't have this — need external solver or Lanczos implementation)

## Key Technical Decisions

### Static vs Dynamic
Static solve is just `K·u = f` (one PCG call, no time stepping). This is much simpler and more efficient than dynamic simulation. For FEA, static is the primary mode.

### Linear vs Nonlinear
- **Linear:** Small-strain assumption, K is constant. One solve. Fast.
- **Nonlinear:** K depends on displacement (geometric nonlinearity). Need Newton-Raphson iteration (multiple PCG solves). We already have this in the dynamic solver.

Start with linear (Hooke's law), add nonlinear as option.

### Stress Recovery
StVK gives us Green-Lagrange strain E and 2nd Piola-Kirchhoff stress S directly. For engineering use:
- Cauchy stress σ = (1/J) F·S·Fᵀ where J = det(F)
- Von Mises σ_vm from deviatoric part of σ
- These are per-element quantities; interpolate to vertices for visualization

### Comparison with Ansys/Abaqus
| Feature | This solver | Ansys |
|---------|-------------|-------|
| Element types | Shell tri + Solid tet | 200+ types |
| Material models | StVK, Neo-Hookean, ElastoPlastic | 100+ models |
| Solver | PCG (GPU) | Direct + iterative (CPU/GPU) |
| Mesh generation | External (Newton ModelBuilder) | Built-in mesher |
| Pre/post processing | USD + Python | Full GUI |
| GPU acceleration | Native (Warp) | Optional (CUDA) |
| Cost | Free/open source | $$$$ |

The key advantage: **native GPU acceleration** via Warp. For problems that fit in GPU memory, this could be significantly faster than CPU-based Ansys for large meshes.

## Dependencies

- Everything from `feature/fem-shell-solver` (SolverFEMShell)
- No new external dependencies for Phase 1-3
- Phase 4 modal analysis might need scipy.sparse.linalg.eigsh or custom Lanczos

## Open Questions

1. **Mesh format:** Support STEP/IGES import? Or require pre-meshed input (USD/OBJ/VTK)?
2. **Validation:** Which standard benchmarks to use? (Scordelis-Lo roof, pinched hemisphere, etc.)
3. **Licensing:** If this becomes a product, what license? Newton is Apache-2.0.
4. **Integration with USD:** Should stress results be stored as USD primvars or separate files?
