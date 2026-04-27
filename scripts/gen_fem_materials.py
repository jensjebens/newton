"""Generate FEM Shell + IPC material comparison GIFs and USD stages.

Same scene as VBD baselines: 20×20 grid draping onto sphere + ground.
Run: cd newton && PYTHONPATH=. python3 scripts/gen_fem_materials.py
"""
import os
import sys
import numpy as np
import warp as wp

wp.init()

import newton
from newton.solvers import SolverFEMShell

# Scene setup — sheet starts just above sphere for quick contact
GRID_DIM = 20
CELL_SIZE = 0.04  # 20×20 × 0.04 = 0.8m × 0.8m
START_Z = 0.35    # just above sphere top (z=0.30)
SPHERE_POS = (0.4, 0.4, 0.15)
SPHERE_RADIUS = 0.15
GROUND_Z = 0.0
N_FRAMES = 180
FPS = 60
SUBSTEPS = 16
DT = 1.0 / FPS

# Materials: name → (E, nu, h, density_kg_m3)
MATERIALS = {
    "cotton":    (5e6,  0.3,  0.0003, 300),
    "silk":      (2e6,  0.3,  0.0001, 100),
    "paper":     (3e9,  0.25, 0.0001, 750),
    "cardboard": (3e9,  0.3,  0.003,  300),
    "pvc_thin":  (1e9,  0.4,  0.0005, 1400),
    "pvc_thick": (3e9,  0.4,  0.003,  1400),
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "materials")
os.makedirs(OUT_DIR, exist_ok=True)


def compute_mass_per_vertex(density, thickness, grid_dim, cell_size):
    """Compute per-vertex mass from physical density, thickness, and grid area."""
    sheet_area = (grid_dim * cell_size) ** 2
    total_mass = density * thickness * sheet_area
    n_verts = (grid_dim + 1) ** 2
    return total_mass / n_verts


def write_usda(filepath, frames, sphere_pos, sphere_radius, grid_dim, n_frames):
    """Write USD stage with time-sampled mesh, sphere, ground, camera."""
    n_verts = (grid_dim + 1) ** 2
    n_faces = grid_dim * grid_dim * 2  # triangulated

    # Build face topology (triangulated grid)
    face_counts = []
    face_indices = []
    for y in range(grid_dim):
        for x in range(grid_dim):
            v00 = y * (grid_dim + 1) + x
            v10 = v00 + 1
            v01 = v00 + (grid_dim + 1)
            v11 = v01 + 1
            # Two triangles per quad
            face_counts.extend([3, 3])
            face_indices.extend([v00, v10, v11, v00, v11, v01])

    with open(filepath, 'w') as f:
        f.write('#usda 1.0\n')
        f.write('(\n    defaultPrim = "World"\n    metersPerUnit = 1\n    upAxis = "Z"\n')
        f.write(f'    startTimeCode = 0\n')
        f.write(f'    endTimeCode = {n_frames}\n')
        f.write(f'    timeCodesPerSecond = 60\n')
        f.write(')\n\n')
        f.write('def Xform "World"\n{\n')

        # Camera
        # Camera: 3/4 view from above looking at sphere center
        f.write('    def Camera "Camera"\n    {\n')
        f.write('        float focalLength = 30\n')
        f.write('        double3 xformOp:translate = (-0.3, -0.8, 1.0)\n')
        f.write('        float3 xformOp:rotateXYZ = (55, 0, -15)\n')
        f.write('        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]\n')
        f.write('    }\n\n')

        # Sphere
        f.write('    def Sphere "Sphere"\n    {\n')
        f.write(f'        double radius = {sphere_radius}\n')
        f.write(f'        double3 xformOp:translate = ({sphere_pos[0]}, {sphere_pos[1]}, {sphere_pos[2]})\n')
        f.write('        uniform token[] xformOpOrder = ["xformOp:translate"]\n')
        f.write('        color3f[] primvars:displayColor = [(0.8, 0.1, 0.1)]\n')
        # Material
        f.write('        def Material "SphereMat"\n        {\n')
        f.write('            token outputs:surface.connect = </World/Sphere/SphereMat/Shader.outputs:surface>\n')
        f.write('            def Shader "Shader"\n            {\n')
        f.write('                uniform token info:id = "UsdPreviewSurface"\n')
        f.write('                color3f inputs:diffuseColor = (0.8, 0.1, 0.1)\n')
        f.write('                token outputs:surface\n')
        f.write('            }\n        }\n')
        f.write('        rel material:binding = </World/Sphere/SphereMat>\n')
        f.write('    }\n\n')

        # Ground plane
        f.write('    def Mesh "GroundPlane"\n    {\n')
        f.write('        int[] faceVertexCounts = [4]\n')
        f.write('        int[] faceVertexIndices = [0, 1, 2, 3]\n')
        f.write('        point3f[] points = [(-1, -1, 0), (2, -1, 0), (2, 2, 0), (-1, 2, 0)]\n')
        f.write('        color3f[] primvars:displayColor = [(0.5, 0.5, 0.5)]\n')
        f.write('        def Material "GroundMat"\n        {\n')
        f.write('            token outputs:surface.connect = </World/GroundPlane/GroundMat/Shader.outputs:surface>\n')
        f.write('            def Shader "Shader"\n            {\n')
        f.write('                uniform token info:id = "UsdPreviewSurface"\n')
        f.write('                color3f inputs:diffuseColor = (0.5, 0.5, 0.5)\n')
        f.write('                token outputs:surface\n')
        f.write('            }\n        }\n')
        f.write('        rel material:binding = </World/GroundPlane/GroundMat>\n')
        f.write('    }\n\n')

        # Cloth mesh with time samples
        f.write('    def Mesh "Cloth"\n    {\n')
        f.write(f'        int[] faceVertexCounts = {face_counts}\n')
        f.write(f'        int[] faceVertexIndices = {face_indices}\n')

        # Time-sampled points
        f.write('        point3f[] points.timeSamples = {\n')
        for frame_idx, (time_code, positions) in enumerate(frames):
            pts_str = ", ".join(f"({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})" for p in positions)
            comma = "," if frame_idx < len(frames) - 1 else ""
            f.write(f'            {time_code}: [{pts_str}]{comma}\n')
        f.write('        }\n')

        f.write('        color3f[] primvars:displayColor = [(0.2, 0.5, 0.8)]\n')
        f.write('        def Material "ClothMat"\n        {\n')
        f.write('            token outputs:surface.connect = </World/Cloth/ClothMat/Shader.outputs:surface>\n')
        f.write('            def Shader "Shader"\n            {\n')
        f.write('                uniform token info:id = "UsdPreviewSurface"\n')
        f.write('                color3f inputs:diffuseColor = (0.2, 0.5, 0.8)\n')
        f.write('                token outputs:surface\n')
        f.write('            }\n        }\n')
        f.write('        rel material:binding = </World/Cloth/ClothMat>\n')
        f.write('    }\n')
        f.write('}\n')


def simulate_material(name, E, nu, h, density):
    """Run FEM shell simulation and return per-frame positions."""
    print(f"\n{'='*60}")
    print(f"Simulating: {name} (E={E:.0e}, nu={nu}, h={h*1000:.1f}mm, rho={density})")
    print(f"{'='*60}")

    mass = compute_mass_per_vertex(density, h, GRID_DIM, CELL_SIZE)
    print(f"  Per-vertex mass: {mass:.6f} kg")

    builder = newton.ModelBuilder(gravity=-9.81)
    builder.add_cloth_grid(
        pos=wp.vec3(0, 0, START_Z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0, 0, 0),
        dim_x=GRID_DIM,
        dim_y=GRID_DIM,
        cell_x=CELL_SIZE,
        cell_y=CELL_SIZE,
        mass=mass,
    )
    builder.color(include_bending=True)
    model = builder.finalize("cuda:0")

    solver = SolverFEMShell(
        model,
        young_modulus=E,
        poisson_ratio=nu,
        thickness=h,
        substeps=SUBSTEPS,
        damping=0.0005,  # minimal damping — let physics drive the motion
        use_ipc=True,
        ipc_dhat=0.01,
    )
    solver.set_ipc_ground(z=GROUND_Z)
    # Also set sphere contact via legacy penalty (IPC sphere not yet implemented)
    solver.set_contact_sphere(SPHERE_POS, SPHERE_RADIUS, stiffness=1e6, damping=100)

    s0, s1 = model.state(), model.state()
    ctrl = model.control()
    contacts = model.contacts()

    # Give initial downward velocity to push through over-damping
    vel_np = s0.particle_qd.numpy()
    vel_np[:, 2] = -2.0  # gentle push toward sphere
    s0.particle_qd = wp.array(vel_np, dtype=wp.vec3, device="cuda:0")

    frames = []
    # Frame 0
    wp.synchronize()
    pos0 = s0.particle_q.numpy().copy()
    frames.append((0, pos0))

    for frame in range(1, N_FRAMES + 1):
        s0.clear_forces()
        model.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, DT)
        s0, s1 = s1, s0

        if frame % 10 == 0 or frame == N_FRAMES:
            wp.synchronize()
            pos = s0.particle_q.numpy()
            z_range = f"z=[{np.min(pos[:,2]):.3f}, {np.max(pos[:,2]):.3f}]"
            print(f"  Frame {frame}/{N_FRAMES}: {z_range}")

        wp.synchronize()
        frames.append((frame, s0.particle_q.numpy().copy()))

    return frames


def render_gif(usda_path, gif_path, n_frames):
    """Render USD to GIF using usdrecord + imageio."""
    import subprocess
    import tempfile

    # Find usdrecord
    usdrecord = os.path.expanduser("~/builds/usd-vanilla-build/bin/usdrecord")
    if not os.path.exists(usdrecord):
        usdrecord = os.path.expanduser("~/builds/usd-fork-clean-build/bin/usdrecord")

    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
        env["PYTHONPATH"] = os.path.expanduser("~/builds/usd-vanilla-build/lib/python")
        env["LD_LIBRARY_PATH"] = os.path.expanduser("~/builds/usd-vanilla-build/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        env["PXR_PLUGINPATH_NAME"] = ""

        # Render every 3rd frame using usdrecord's frame range
        out_pattern = os.path.join(tmpdir, "frame_####.png")
        step = 3
        frame_spec = f"0:{n_frames}x{step}"
        cmd = [
            sys.executable, usdrecord,
            "--frames", frame_spec,
            "--imageWidth", "640",
            "--camera", "/World/Camera",
            usda_path,
            out_pattern,
        ]
        result = subprocess.run(cmd, env=env, capture_output=True, timeout=300)
        if result.returncode != 0:
            print(f"  usdrecord error: {result.stderr.decode()[-200:]}")

        # Collect rendered frames
        frame_paths = sorted([
            os.path.join(tmpdir, f) for f in os.listdir(tmpdir)
            if f.startswith("frame_") and f.endswith(".png")
        ])

        if not frame_paths:
            print(f"  WARNING: usdrecord produced no frames for {usda_path}")
            return False

        # Combine to GIF with imageio
        try:
            import imageio.v2 as imageio
            images = [imageio.imread(p) for p in frame_paths]
            imageio.mimsave(gif_path, images, duration=1.0/20, loop=0)
            size_kb = os.path.getsize(gif_path) / 1024
            print(f"  GIF: {gif_path} ({len(images)} frames, {size_kb:.0f} KB)")
            return True
        except ImportError:
            print("  WARNING: imageio not installed, skipping GIF")
            return False


if __name__ == "__main__":
    results = {}

    for name, (E, nu, h, density) in MATERIALS.items():
        try:
            frames = simulate_material(name, E, nu, h, density)
            results[name] = frames

            # Write USD
            usda_path = os.path.join(OUT_DIR, f"{name}_fem.usda")
            write_usda(usda_path, frames, SPHERE_POS, SPHERE_RADIUS, GRID_DIM, N_FRAMES)
            size_mb = os.path.getsize(usda_path) / (1024 * 1024)
            print(f"  USD: {usda_path} ({size_mb:.1f} MB)")

        except Exception as e:
            print(f"  ERROR: {name} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"All simulations complete. Rendering GIFs...")
    print(f"{'='*60}")

    for name in results:
        usda_path = os.path.join(OUT_DIR, f"{name}_fem.usda")
        gif_path = os.path.join(OUT_DIR, f"{name}_fem.gif")
        print(f"\nRendering {name}...")
        render_gif(usda_path, gif_path, N_FRAMES)

    print("\nDone!")
