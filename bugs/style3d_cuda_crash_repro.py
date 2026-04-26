"""
Minimal reproduction: Style3D CUDA crash on L40

Two bugs found:
  BUG 1: np.atan2 doesn't exist in numpy (should be np.arctan2)
         - File: newton/_src/solvers/style3d/cloth.py line 146
         - Affects: Newton 1.0.0 and 1.1.0
         - Prevents Style3D cloth model construction entirely
         
  BUG 2: CUDA error 700 (illegal memory access) in wp_array_inner_float_device
         - File: newton/_src/solvers/style3d/linear_solver.py line 235
         - Triggered by PCG solver's array_inner() ctypes FFI call
         - Warp reduce kernel (reduce.cu:239) crash on L40

Environment:
  GPU: NVIDIA L40 (sm_89, 48GB)
  Driver: 570.158.01 (CUDA runtime 12.8)
  Warp: 1.12.0 (bundles CUDA Toolkit 12.9)
  Newton: 1.1.0 (also 1.0.0)
  numpy: 1.26.4
  Python: 3.10.12
  OS: Ubuntu 22.04.5 LTS
"""

import sys
import traceback
import numpy as np

# Monkey-patch BUG 1: numpy API misuse in Newton's Style3D cloth.py
# np.atan2 should be np.arctan2, np.pow should be np.power
patched = []
if not hasattr(np, 'atan2'):
    np.atan2 = np.arctan2
    patched.append('np.atan2 -> np.arctan2')
if not hasattr(np, 'pow'):
    np.pow = np.power
    patched.append('np.pow -> np.power')
if patched:
    print(f"[PATCHED] Bug 1 workarounds: {', '.join(patched)}")

import warp as wp
wp.init()

print(f"\nWarp {wp.__version__}")
print(f"Device: {wp.get_device('cuda:0')}")
print()

import newton
from newton.solvers import style3d
SolverStyle3D = style3d.solver_style3d.SolverStyle3D

print("=== Building minimal cloth scene with Style3D solver ===")

builder = newton.ModelBuilder()
SolverStyle3D.register_custom_attributes(builder)
builder.add_ground_plane()

# Small 8x8 cloth grid — minimal scene
style3d.add_cloth_grid(
    builder,
    pos=wp.vec3(0.0, 0.0, 4.0),
    rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
    vel=wp.vec3(0.0, 0.0, 0.0),
    dim_x=8,
    dim_y=8,
    cell_x=0.1,
    cell_y=0.1,
    mass=0.1,
    fix_left=True,
    tri_aniso_ke=wp.vec3(1.0e4, 1.0e4, 1.0e3),
    edge_aniso_ke=wp.vec3(2.0e-6, 1.0e-6, 5.0e-6),
    particle_radius=0.05,
)

model = builder.finalize("cuda:0")
print(f"Model: {model.particle_count} particles, {model.tri_count} triangles")

print("\n=== Creating Style3D solver ===")
solver = SolverStyle3D(model, iterations=5)

state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()

dt = 1.0 / 120.0

print("\n=== Simulating (5 steps) ===")
for step in range(5):
    try:
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0
        wp.synchronize()
        print(f"  Step {step+1}: OK")
    except Exception as e:
        print(f"\n  Step {step+1}: CRASH!")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error: {e}")
        traceback.print_exc()
        sys.exit(1)

print("\nAll steps completed without crash (Bug 2 NOT reproduced)")
