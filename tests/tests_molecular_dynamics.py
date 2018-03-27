import unittest
import numpy as np
import numpy.testing as nptest
import lib.molecular_dynamics as md


class TestState(unittest.TestCase):

    def test_init(self):
        test_state = md.State(10)  # Default init test
        defaults = np.zeros((3, 10))
        nptest.assert_array_equal(test_state.positions, defaults)
        nptest.assert_array_equal(test_state.velocities, defaults)

    def test_forces(self):
        def force_sho(k, positions):
            return -k*positions
        test_state = md.State(3, 10).init_random((-1, 1), (-1, 1))
        forces = test_state.forces(lambda pos: force_sho(1, pos))
        nptest.assert_array_almost_equal(forces, test_state.positions*-1)

    def test_integrator_step(self):
        no_force = lambda positions: 0
        constant_force = lambda positions: 1
        sho_foce = lambda positions: -positions
        init_state = md.State(1, dim=1)
        dt = 1
        integrator = md.VerletIntegrator(init_state, no_force, dt)
        next_step = next(integrator)
        nptest.assert_array_equal(next_step.positions, md.State(1, dim=1).positions)  # For no force, state remains
        nptest.assert_array_equal(next_step.velocities, md.State(1, dim=1).velocities)

        integrator = md.VerletIntegrator(init_state, constant_force, dt)
        next_step = next(integrator)
        after_constant = md.State(1, dim=1)
        after_constant.positions = np.array([[0.5]])
        after_constant.velocities = np.array([[1]])
        nptest.assert_array_almost_equal(next_step.positions, after_constant.positions)
        nptest.assert_array_almost_equal(next_step.velocities, after_constant.velocities)
