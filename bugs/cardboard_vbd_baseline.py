"""
Cardboard Simulation Baseline — Newton VBD Solver
Phase 6 "before" comparison: VBD cloth solver with high stiffness parameters
tuned for cardboard-like material behavior.

This serves as the baseline for comparing against the Warp FEM + IPC solver.
"""

import sys
import numpy as np
import warp as wp

wp.init()

import newton
from newton.solvers import SolverVBD

print("=" * 60)
print("CARDBOARD SIMULATION BASELINE — Newton VBD")
print("=" * 60)

# ========================================
# Material parameters for cardboard
# ========================================
# Real cardboard: E ~ 1-5 GPa, ν ~ 0.3, h ~ 2mm, ρ ~ 600 kg/m³
# In Newton's PBD/VBD params:
#   tri_ke  = elastic stiffness (stretch resistance) — HIGH for cardboard
#   tri_ka  = area stiffness — HIGH (cardboard doesn't stretch)
#   tri_kd  = damping
#   edge_ke = bending stiffness — HIGH (cardboard is stiff)

CARDBOARD_TRI_KE = 1.0e5    # Very high stretch resistance
CARDBOARD_TRI_KA = 1.0e5    # Very high area preservation
CARDBOARD_TRI_KD = 10.0     # Some damping
CARDBOARD_EDGE_KE = 1.0e4   # High bending stiffness (this is what makes it "cardboard" vs "cloth")
CARDBOARD_EDGE_KD = 1.0     # Edge damping
CARDBOARD_MASS = 0.05       # Mass per particle (light-ish)
CARDBOARD_RADIUS = 0.02     # Particle radius for collision

# Grid size
DIM_X = 16
DIM_Y = 16
CELL_SIZE = 0.05  # 5cm cells → 80cm × 80cm sheet

print(f"\nSheet: {DIM_X}x{DIM_Y} grid, {CELL_SIZE*100:.0f}cm cells")
print(f"  Total size: {DIM_X*CELL_SIZE:.2f}m × {DIM_Y*CELL_SIZE:.2f}m")
print(f"  Particles: {(DIM_X+1)*(DIM_Y+1)}")
print(f"  Triangles: {DIM_X*DIM_Y*2}")
print(f"\nMaterial params (VBD):")
print(f"  tri_ke (stretch): {CARDBOARD_TRI_KE:.0e}")
print(f"  tri_ka (area):    {CARDBOARD_TRI_KA:.0e}")
print(f"  edge_ke (bend):   {CARDBOARD_EDGE_KE:.0e}")

# ========================================
# Build scene
# ========================================
builder = newton.ModelBuilder()
builder.add_ground_plane()

# Cardboard sheet — suspended from one edge, falls under gravity
builder.add_cloth_grid(
    pos=wp.vec3(0.0, 0.0, 1.5),  # 1.5m above ground (Z-up in Newton)
    rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi * 0.0),  # flat
    vel=wp.vec3(0.0, 0.0, 0.0),
    dim_x=DIM_X,
    dim_y=DIM_Y,
    cell_x=CELL_SIZE,
    cell_y=CELL_SIZE,
    mass=CARDBOARD_MASS,
    fix_left=True,  # Pin left edge (cantilever)
    tri_ke=CARDBOARD_TRI_KE,
    tri_ka=CARDBOARD_TRI_KA,
    tri_kd=CARDBOARD_TRI_KD,
    edge_ke=CARDBOARD_EDGE_KE,
    edge_kd=CARDBOARD_EDGE_KD,
    particle_radius=CARDBOARD_RADIUS,
)

# VBD requires graph coloring for parallel Gauss-Seidel
builder.color(include_bending=True)

model = builder.finalize("cuda:0")
print(f"\nModel finalized: {model.particle_count} particles, {model.tri_count} triangles")

# ========================================
# Create VBD solver
# ========================================
print("\nCreating VBD solver...")
solver = SolverVBD(model, iterations=20)

state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()

# ========================================
# Simulate
# ========================================
fps = 60
frame_dt = 1.0 / fps
substeps = 4
sim_dt = frame_dt / substeps
num_frames = 120  # 2 seconds

print(f"\nSimulating {num_frames} frames @ {fps}fps ({substeps} substeps/frame)")
print(f"  dt = {sim_dt:.4f}s")

# Track some metrics
positions_history = []
max_sag = []

for frame in range(num_frames):
    for sub in range(substeps):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, sim_dt)
        state_0, state_1 = state_1, state_0

    # Record particle positions every 10 frames
    if frame % 10 == 0 or frame == num_frames - 1:
        wp.synchronize()
        pos = state_0.particle_q.numpy()
        z_vals = pos[:, 2]  # Z is up in Newton
        sag = 1.5 - np.min(z_vals)
        max_sag.append(sag)
        
        # Find the free end (rightmost particles)
        # In our grid, rightmost column is at x = DIM_X * CELL_SIZE
        right_col = pos[DIM_X::DIM_X+1]  # every (DIM_X+1)th starting from DIM_X
        free_end_z = np.mean(right_col[:, 2]) if len(right_col) > 0 else 0
        
        print(f"  Frame {frame:3d}: max_sag={sag:.4f}m, free_end_z={free_end_z:.4f}m")

wp.synchronize()

# ========================================
# Final state analysis
# ========================================
final_pos = state_0.particle_q.numpy()
print(f"\n{'=' * 60}")
print("BASELINE RESULTS")
print(f"{'=' * 60}")
print(f"  Final max sag:     {max_sag[-1]:.4f}m")
print(f"  Z range:           [{np.min(final_pos[:, 2]):.4f}, {np.max(final_pos[:, 2]):.4f}]m")
print(f"  Stretch (X range): [{np.min(final_pos[:, 0]):.4f}, {np.max(final_pos[:, 0]):.4f}]m")
print(f"  Expected X range:  [0, {DIM_X * CELL_SIZE:.4f}]m")

x_range = np.max(final_pos[:, 0]) - np.min(final_pos[:, 0])
expected_x = DIM_X * CELL_SIZE
stretch_ratio = x_range / expected_x
print(f"  Stretch ratio:     {stretch_ratio:.4f} (1.0 = no stretch)")

# Save final positions for comparison
np.save("/tmp/cardboard_vbd_baseline.npy", final_pos)
print(f"\nFinal positions saved to /tmp/cardboard_vbd_baseline.npy")
print(f"\nBaseline complete. This is the 'before' for FEM+IPC comparison.")
