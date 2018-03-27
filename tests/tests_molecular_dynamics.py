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
