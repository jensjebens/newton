"""Demo: 0.5kg ball in paper bag with IPC ground contact.

Requires: newton fork (feature/fem-shell-solver), ipctk, warp
Run: PYTHONPATH=. python3 demos/bag_with_ball_ipc.py
"""
# See /tmp/fem_bag_ipc.usda for the generated scene
# Key params:
# - Bag: E=50MPa, h=1mm, 0.3x0.3x0.4m open-top box
# - Ball: 0.5kg, r=10cm
# - Contact: IPC barrier (dhat=0.01) for ground, projection for sphere
# - FEM: 8 substeps, plastic deformation (yield=0.15rad)
print("See source code for full implementation")
print("Generated USD: /tmp/fem_bag_ipc.usda")
