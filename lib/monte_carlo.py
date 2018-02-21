import numpy as np
import matplotlib.pyplot as plt
import lib.util as util
import unittest


class MonteCarlo(object):
    """Base Monte Carlo simulator, defines virtual interface"""
    def __init__(self, num_runs):
        self._num_runs = num_runs

    def init_state(self):
        pass

    def plot_state(self, *args, **kwargs):
        pass

    def variable_error(self):
        pass


class IsingModel(MonteCarlo):
    """Ising model solver using Monte Carlo method"""
    def __init__(self, num_runs, magnetic_field, lattice_side):
        super().__init__(num_runs)
        self._magnetic_field = magnetic_field
        self._lattice_side = lattice_side


class SpinConfiguration(object):
    """Object representing spins on a lattice"""
    def __init__(self, lattice):
        self._lattice = lattice

    @classmethod
    def all_up(cls, rows, columns):
        """Initialize configuration with all spins up"""
        return cls(np.ones((rows, columns)))

    @classmethod
    def all_down(cls, rows, columns):
        """Initialize configuration with all spins down"""
        return cls(np.ones((rows, columns))*-1)

    @classmethod
    def init_random(cls, rows, columns):
        """Initialize a random spin configuration in the desired shape"""
        return None

    def __setitem__(self, row, column, value):
        """Set a spin to a value"""
        pass

    def __getitem__(self, row, column):
        pass

    def magnetization(self):
        """Returns magnetization of the lattice"""
        return np.sum(self._lattice)

    def flip_spin(self, row, column):
        pass


class SpinConfigTest(unittest.TestCase):
    """Tests for SpinConfiguration class"""
    def test_initialize(self):
        some_lattice = np.array([[1, -1], [-1, -1]])
        config = SpinConfiguration(some_lattice)
        config_up = SpinConfiguration.all_up(5, 5)
        config_down = SpinConfiguration.all_down(5, 5)
        np.testing.assert_array_equal(config._lattice, some_lattice)
        np.testing.assert_array_equal(config_up._lattice, np.ones((5, 5)))
        np.testing.assert_array_equal(config_down._lattice, np.ones((5, 5))*-1)
        random_lattice = SpinConfiguration.init_random(10, 10)
        SpinConfiguration(random_lattice._lattice)  # Check if init_random returns a valid lattice
        with self.assertRaises(ValueError):
            SpinConfiguration(np.array([[1, 2], [-1, 1]]))

    def test_magnetization(self):
        some_lattice = np.array([[1, -1], [-1, -1]])
        config = SpinConfiguration(some_lattice)
        magnetization = np.sum(some_lattice)
        self.assertEqual(config.magnetization(), magnetization)


if __name__ == '__main__':
    unittest.main()
