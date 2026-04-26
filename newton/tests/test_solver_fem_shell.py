# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FEM thin shell solver — Phase A (membrane only).

These tests define the expected behavior of the SolverFEMShell solver.
They must FAIL initially (solver doesn't exist yet) and PASS after implementation.
"""

import unittest

import numpy as np
import warp as wp


# ---------------------------------------------------------------------------
# Test 1: Solver class exists and follows Newton's solver API
# ---------------------------------------------------------------------------
class TestSolverFEMShellAPI(unittest.TestCase):
    """SolverFEMShell must inherit from SolverBase and follow the step() API."""

    def test_import(self):
        """SolverFEMShell must be importable from newton.solvers."""
        from newton.solvers import SolverFEMShell  # noqa: F401

    def test_inherits_solver_base(self):
        """SolverFEMShell must inherit from SolverBase."""
        from newton._src.solvers.solver import SolverBase
        from newton.solvers import SolverFEMShell

        self.assertTrue(issubclass(SolverFEMShell, SolverBase))

    def test_step_signature(self):
        """step() must accept (state_0, state_1, control, contacts, dt)."""
        import inspect

        from newton.solvers import SolverFEMShell

        sig = inspect.signature(SolverFEMShell.step)
        params = list(sig.parameters.keys())
        # self, state_0, state_1, control, contacts, dt
        self.assertGreaterEqual(len(params), 6)

    def test_constructor_accepts_model(self):
        """Constructor must accept a Newton Model."""
        from newton.solvers import SolverFEMShell

        wp.init()
        import newton

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 0, 1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=4,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
        )
        builder.color(include_bending=True)
        model = builder.finalize("cuda:0")

        # Should not raise
        solver = SolverFEMShell(model)
        self.assertIsNotNone(solver)


# ---------------------------------------------------------------------------
# Test 2: Physical material parameters
# ---------------------------------------------------------------------------
class TestPhysicalMaterialParams(unittest.TestCase):
    """SolverFEMShell must accept physical material parameters (E, nu, h, rho)."""

    def test_accepts_youngs_modulus(self):
        """Constructor must accept young_modulus parameter."""
        from newton.solvers import SolverFEMShell

        wp.init()
        import newton

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 0, 1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=4,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
        )
        builder.color(include_bending=True)
        model = builder.finalize("cuda:0")

        # Cardboard: E=3 GPa, nu=0.3, h=2mm
        solver = SolverFEMShell(
            model,
            young_modulus=3.0e9,
            poisson_ratio=0.3,
            thickness=0.002,
        )
        self.assertAlmostEqual(solver.young_modulus, 3.0e9)
        self.assertAlmostEqual(solver.poisson_ratio, 0.3)
        self.assertAlmostEqual(solver.thickness, 0.002)


# ---------------------------------------------------------------------------
# Test 3: Membrane inextensibility under gravity
# ---------------------------------------------------------------------------
class TestMembraneInextensibility(unittest.TestCase):
    """A sheet under gravity should resist stretching (< 1% for stiff material)."""

    def test_stiff_membrane_no_stretch(self):
        """Stiff membrane (E=3GPa) should have < 1% stretch after 1 second."""
        from newton.solvers import SolverFEMShell

        wp.init()
        import newton

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 0, 1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=10,
            dim_y=10,
            cell_x=0.08,
            cell_y=0.08,  # 0.8m x 0.8m
            mass=0.05,
            fix_left=True,  # cantilever
        )
        builder.color(include_bending=True)
        model = builder.finalize("cuda:0")

        solver = SolverFEMShell(
            model,
            young_modulus=3.0e9,  # cardboard
            poisson_ratio=0.3,
            thickness=0.002,
        )

        s0, s1 = model.state(), model.state()
        ctrl = model.control()
        contacts = model.contacts()

        # Record initial width
        pos0 = s0.particle_q.numpy()
        initial_width = np.max(pos0[:, 0]) - np.min(pos0[:, 0])

        # Simulate 1 second
        dt = 1.0 / 60.0
        for _ in range(60):
            s0.clear_forces()
            model.collide(s0, contacts)
            solver.step(s0, s1, ctrl, contacts, dt)
            s0, s1 = s1, s0

        wp.synchronize()
        pos_final = s0.particle_q.numpy()
        final_width = np.max(pos_final[:, 0]) - np.min(pos_final[:, 0])

        stretch_ratio = abs(final_width - initial_width) / initial_width
        self.assertLess(stretch_ratio, 0.01, f"Stretch {stretch_ratio:.3%} exceeds 1% for stiff membrane")


# ---------------------------------------------------------------------------
# Test 4: Soft membrane allows deformation
# ---------------------------------------------------------------------------
class TestSoftMembraneDeforms(unittest.TestCase):
    """A soft membrane should deform under gravity (not stay rigid)."""

    def test_soft_membrane_sags(self):
        """Soft membrane (E=1MPa) cantilever should sag noticeably (> 5cm in 1s)."""
        from newton.solvers import SolverFEMShell

        wp.init()
        import newton

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 0, 1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=10,
            dim_y=10,
            cell_x=0.08,
            cell_y=0.08,
            mass=0.05,
            fix_left=True,
        )
        builder.color(include_bending=True)
        model = builder.finalize("cuda:0")

        solver = SolverFEMShell(
            model,
            young_modulus=1.0e6,  # soft rubber
            poisson_ratio=0.3,
            thickness=0.002,
        )

        s0, s1 = model.state(), model.state()
        ctrl = model.control()
        contacts = model.contacts()

        pos0 = s0.particle_q.numpy()
        initial_z_min = np.min(pos0[:, 2])

        dt = 1.0 / 60.0
        for _ in range(60):
            s0.clear_forces()
            model.collide(s0, contacts)
            solver.step(s0, s1, ctrl, contacts, dt)
            s0, s1 = s1, s0

        wp.synchronize()
        pos_final = s0.particle_q.numpy()
        sag = initial_z_min - np.min(pos_final[:, 2])

        self.assertGreater(sag, 0.05, f"Sag {sag:.3f}m is too small for soft membrane")


# ---------------------------------------------------------------------------
# Test 5: Energy conservation (no explosion)
# ---------------------------------------------------------------------------
class TestNoExplosion(unittest.TestCase):
    """Simulation must remain stable — no NaN, no explosion."""

    def test_no_nan_after_simulation(self):
        """Positions must not contain NaN after 2 seconds of simulation."""
        from newton.solvers import SolverFEMShell

        wp.init()
        import newton

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 0, 1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=8,
            dim_y=8,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.05,
        )
        builder.color(include_bending=True)
        model = builder.finalize("cuda:0")

        solver = SolverFEMShell(
            model,
            young_modulus=3.0e9,
            poisson_ratio=0.3,
            thickness=0.002,
        )

        s0, s1 = model.state(), model.state()
        ctrl = model.control()
        contacts = model.contacts()

        dt = 1.0 / 60.0
        for _ in range(120):  # 2 seconds
            s0.clear_forces()
            model.collide(s0, contacts)
            solver.step(s0, s1, ctrl, contacts, dt)
            s0, s1 = s1, s0

        wp.synchronize()
        pos = s0.particle_q.numpy()

        self.assertFalse(np.any(np.isnan(pos)), "Positions contain NaN — simulation exploded")
        self.assertFalse(np.any(np.abs(pos) > 100), "Positions > 100m — simulation exploded")


# ---------------------------------------------------------------------------
# Test 6: Implicit solve convergence
# ---------------------------------------------------------------------------
class TestImplicitSolve(unittest.TestCase):
    """The PCG solver must converge for stiff materials."""

    def test_pcg_converges(self):
        """PCG solver residual must decrease below tolerance."""
        from newton.solvers import SolverFEMShell

        wp.init()
        import newton

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 0, 1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=8,
            dim_y=8,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.05,
        )
        builder.color(include_bending=True)
        model = builder.finalize("cuda:0")

        solver = SolverFEMShell(
            model,
            young_modulus=3.0e9,
            poisson_ratio=0.3,
            thickness=0.002,
        )

        s0, s1 = model.state(), model.state()
        ctrl = model.control()
        contacts = model.contacts()

        # Single step — should converge
        s0.clear_forces()
        model.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, 1.0 / 60.0)

        # Solver should expose convergence info
        self.assertTrue(hasattr(solver, "last_residual"), "Solver must expose last_residual")
        self.assertLess(solver.last_residual, 1e-4, f"PCG didn't converge: residual={solver.last_residual}")


if __name__ == "__main__":
    wp.init()
    unittest.main()
