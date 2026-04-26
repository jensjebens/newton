# Material Simulation Reference — VBD Cloth Drape Test

> **Newton Physics Engine** · VBD (Vertex Block Descent) Solver  
> 6 materials simulated as cloth draping onto a static sphere  
> Generated: April 2025

## Scene Setup

| Parameter | Value |
|-----------|-------|
| **Grid** | 20×20 vertices (0.8m × 0.8m) |
| **Start height** | z = 0.8m |
| **Sphere** | radius = 0.15m, position = (0.4, 0.4, 0.15) |
| **Ground** | z = 0 |
| **Frames** | 180 @ 60fps (3 seconds) |
| **Substeps** | 8 per frame |
| **VBD iterations** | 15 |
| **Gravity** | -9.81 m/s² (Z-down) |
| **Coordinate system** | Z-up |

---

## Material Parameters

### VBD Simulation Parameters

| Material | tri_ke | tri_ka | tri_kd | edge_ke | edge_kd | mass (kg) |
|----------|--------|--------|--------|---------|---------|-----------|
| Cotton | 5,000 | 5,000 | 5.0 | 100 | 0.5 | 0.030 |
| Silk | 1,000 | 1,000 | 8.0 | 5 | 1.0 | 0.005 |
| Paper | 50,000 | 50,000 | 5.0 | 500 | 1.0 | 0.015 |
| Cardboard | 100,000 | 100,000 | 10.0 | 10,000 | 1.0 | 0.050 |
| PVC Thin (0.5mm) | 200,000 | 200,000 | 8.0 | 2,000 | 2.0 | 0.030 |
| PVC Thick (3mm) | 500,000 | 500,000 | 15.0 | 50,000 | 3.0 | 0.080 |

**Parameter key:**
- `tri_ke` — Triangle elastic stiffness (stretch resistance)
- `tri_ka` — Triangle area stiffness (area preservation)
- `tri_kd` — Triangle damping
- `edge_ke` — Edge/bending stiffness (higher = more rigid)
- `edge_kd` — Edge/bending damping
- `mass` — Per-vertex mass in kg

### Real-World Physical Properties (Reference)

| Material | Young's Modulus | Density | Typical Thickness | Poisson's Ratio |
|----------|----------------|---------|-------------------|-----------------|
| Cotton (woven) | 5–13 GPa (fiber), ~1–10 MPa (fabric) | 1,500 kg/m³ (fiber), ~200–400 g/m² (fabric) | 0.2–0.5 mm | 0.3–0.4 |
| Silk | 5–12 GPa (fiber), ~0.5–5 MPa (fabric) | 1,300 kg/m³ (fiber), ~50–100 g/m² (fabric) | 0.05–0.15 mm | 0.3 |
| Paper (office) | 1–10 GPa | 700–800 kg/m³ | 0.05–0.1 mm | 0.2–0.3 |
| Cardboard (corrugated) | 2–5 GPa | 150–400 kg/m³ | 1–4 mm | 0.2–0.35 |
| PVC (flexible, 0.5mm) | 0.01–1.5 GPa | 1,300–1,450 kg/m³ | 0.5 mm | 0.38–0.42 |
| PVC (rigid, 3mm) | 2.4–4.1 GPa | 1,300–1,450 kg/m³ | 3 mm | 0.38–0.42 |

> **Note:** VBD parameters are not direct physical quantities. They're solver-specific stiffness/damping values tuned to approximate observed material behavior. The real-world table is provided for context when mapping simulation to physically-based materials.

---

## Simulation Results

### 1. Cotton
*Soft, medium weight, low bending stiffness*

![cotton](cotton_vbd.gif)

**Behavior:** The cotton cloth falls smoothly and drapes conformally over the sphere. It shows natural soft folds and the edges hang down loosely. The cloth settles completely within ~2 seconds with gentle oscillation. This is the most "cloth-like" result and closest to what you'd expect from a light fabric.

**VBD verdict:** ✅ Excellent — VBD handles soft cloth very well.

---

### 2. Silk
*Very light, very soft, almost no bending resistance*

![silk](silk_vbd.gif)

**Behavior:** Silk is the lightest and softest material. It drapes very tightly over the sphere with minimal resistance to bending. The low mass and high damping help it settle quickly and conform closely to the sphere. Compared to cotton, silk shows noticeably less structure — the edges hang more limply and the cloth clings rather than folds. Some residual bounce-back from VBD's rest shape constraints is visible but dampened.

**VBD verdict:** ✅ Good — significantly softer than cotton, though VBD's elastic rest shape prevents fully fluid silk behavior. Higher damping (tri_kd=8, edge_kd=1) helps suppress oscillation.

---

### 3. Paper
*Light, moderate stretch resistance, low bending*

![paper](paper_vbd.gif)

**Behavior:** Paper shows moderate stiffness — it doesn't drape as tightly as cotton or silk but still conforms to the sphere's shape. The edges hold some shape rather than hanging limply. There's a visible difference from fabrics: the surface remains smoother with fewer fine wrinkles, and the folds are broader.

**VBD verdict:** ✅ Good — paper behavior is plausible, though real paper would show more creasing/crumpling on contact.

---

### 4. Cardboard
*Stiff, high bending and stretch resistance*

![cardboard](cardboard_vbd.gif)

**Behavior:** Cardboard is significantly stiffer. The sheet bends rather than drapes, maintaining more of its flat shape. The edges don't hang vertically. However, VBD still treats this as a cloth-like material — real cardboard would exhibit plastic deformation (permanent creases) and potentially snap rather than continuously bend.

**VBD verdict:** ⚠️ Limited — VBD produces a stiff but still elastic response. Cardboard's real behavior (plastic deformation, creasing, tearing) cannot be captured with VBD's elastic model. The result looks more like a stiff rubber sheet than actual cardboard.

---

### 5. PVC Thin (0.5mm)
*Flexible plastic sheet*

![pvc_thin](pvc_thin_vbd.gif)

**Behavior:** Thin PVC shows a balance between rigidity and flexibility. It drapes over the sphere but with broader, smoother curvature than fabric. The sheet holds more structural integrity — edges curve rather than hang. The overall shape is plausible for a thin flexible plastic.

**VBD verdict:** ✅ Good — thin PVC behavior is reasonable. VBD's elastic model works acceptably for flexible plastics that don't plastically deform.

---

### 6. PVC Thick (3mm)
*Stiff plastic sheet*

![pvc_thick](pvc_thick_vbd.gif)

**Behavior:** Thick PVC is the stiffest material in the test. The sheet barely deforms — it essentially falls as a near-rigid plate, makes contact with the sphere, and settles with minimal bending. The simulation quickly reaches equilibrium because the sheet cannot flex enough to drape.

**VBD verdict:** ⚠️ Limited — at this stiffness level, the cloth model breaks down. Real 3mm PVC would require a shell/plate formulation to accurately capture bending mechanics. The VBD result is physically plausible (stiff sheet balancing on sphere) but doesn't capture the nuanced behavior of thick plastics.

---

## VBD Solver Notes

### What VBD Does Well
- **Soft fabrics** (cotton, silk): Excellent draping with natural folds and wrinkle patterns
- **Flexible sheets** (paper, thin PVC): Reasonable stiff-but-flexible behavior
- **Collision handling**: Robust cloth-sphere contact without penetration
- **Bending**: With `include_bending=True`, edge-based bending produces realistic fold resistance
- **Performance**: GPU-accelerated, 180 frames × 8 substeps × 15 iterations runs in ~20s on NVIDIA L40

### What VBD Cannot Do
- **Plastic deformation**: No permanent creasing (cardboard, paper folding)
- **Shell mechanics**: True plate/shell bending for thick materials requires FEM formulation
- **Tearing/fracture**: Not supported in VBD cloth model
- **Anisotropy**: Woven fabric directionality (warp/weft) not captured
- **Self-contact at high stiffness**: Very stiff materials may exhibit artifacts

### Parameter Tuning Guidelines
- `tri_ke` / `tri_ka` are the primary stiffness controls — keep them equal for isotropic behavior
- `edge_ke` drives bending resistance — this is the main differentiator between cloth and plate-like behavior
- `mass` significantly affects drape speed and inertia — lighter materials settle faster
- `tri_kd` / `edge_kd` are damping terms — increase to reduce oscillation
- Always call `builder.color(include_bending=True)` to enable bending constraints

---

## FEM Shell + IPC Comparison

> **TODO:** This section will be filled when the FEM Shell solver with IPC (Incremental Potential Contact) is implemented.

### Planned comparisons:
- Same 6 materials with FEM Shell elements
- Direct GIF comparison: VBD vs FEM Shell
- Quantitative metrics:
  - Final drape shape error vs analytical solutions
  - Energy conservation
  - Convergence behavior
  - Bending accuracy for thick materials
- Expected improvements:
  - Cardboard/thick PVC: proper shell bending mechanics
  - Paper: potential for plastic deformation modeling
  - All materials: more physically-based parameter mapping (Young's modulus → shell stiffness)

### Target parameters for FEM Shell:
| Material | Young's Modulus (Pa) | Poisson's Ratio | Thickness (m) | Density (kg/m³) |
|----------|---------------------|-----------------|---------------|-----------------|
| Cotton | 5e6 | 0.3 | 0.0003 | 300 |
| Silk | 2e6 | 0.3 | 0.0001 | 100 |
| Paper | 3e9 | 0.25 | 0.0001 | 750 |
| Cardboard | 3e9 | 0.3 | 0.003 | 300 |
| PVC Thin | 1e9 | 0.4 | 0.0005 | 1400 |
| PVC Thick | 3e9 | 0.4 | 0.003 | 1400 |

---

## Files

| File | Description |
|------|-------------|
| `cotton_vbd.gif` | Cotton drape animation |
| `cotton_vbd.usda` | Cotton simulation USD (180 time samples) |
| `silk_vbd.gif` | Silk drape animation |
| `silk_vbd.usda` | Silk simulation USD |
| `paper_vbd.gif` | Paper drape animation |
| `paper_vbd.usda` | Paper simulation USD |
| `cardboard_vbd.gif` | Cardboard drape animation |
| `cardboard_vbd.usda` | Cardboard simulation USD |
| `pvc_thin_vbd.gif` | PVC Thin (0.5mm) drape animation |
| `pvc_thin_vbd.usda` | PVC Thin simulation USD |
| `pvc_thick_vbd.gif` | PVC Thick (3mm) drape animation |
| `pvc_thick_vbd.usda` | PVC Thick simulation USD |

All USD files contain:
- `/World/Camera` — Perspective camera (focal=35mm)
- `/World/Sphere` — Static red sphere (UsdPreviewSurface, diffuseColor=0.8,0.1,0.1)
- `/World/Cloth` — Animated mesh with per-frame vertex positions
- `/World/GroundPlane` — Ground quad at z=0
