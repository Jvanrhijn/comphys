import numpy as np
import copy
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
        if np.logical_or(np.equal(np.ones(lattice.shape), lattice), np.equal(np.ones(lattice.shape)*-1, lattice)).all():
            self._lattice = copy.deepcopy(lattice)
            self._rows = lattice.shape[0]
            self._columns = lattice.shape[1]
        else:
            raise ValueError("All spins must be either up (1) or down (-1)")

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
        return cls(np.random.choice([1, -1], size=(rows, columns)))

    def __setitem__(self, row, column, value):
        """Set a spin to a value"""
        assert(value in [-1, 1])
        self._lattice[row, column] = value

    def __getitem__(self, key):
        """Get the spin of site (row, column)"""
        return self._lattice[key]

    def magnetization(self):
        """Returns magnetization of the lattice"""
        return np.sum(self._lattice)

    def energy(self, magnetic_field, coupling):
        """Returns the total energy of the lattice"""
        magnetic_energy = -magnetic_field*self.magnetization()
        exchange_energy = 0
        if coupling != 0:
            exchange_energy = self._exchange_energy(coupling)
        return magnetic_energy + exchange_energy

    def _exchange_energy(self, coupling):
        """Return exchange energy of the lattice"""
        exchange_energy = 0
        for i in range(0, self._rows):
            for j in range(0, self._columns):
                # Interaction energy with periodic BC, don't need to modulo for negative indices since in Python
                # list[-1] == list[len(list)-1].
                interaction = self._lattice[i-1, j] \
                              + self._lattice[(i+1) % self._rows, j] \
                              + self._lattice[i, j-1] \
                              + self._lattice[i, (j+1) % self._columns]
                exchange_energy += -coupling * self._lattice[i, j]*interaction
        return exchange_energy

    def flip_spin(self, row, column):
        """Flip a given spin in the lattice"""
        self._lattice[row, column] *= -1

    def flip_random(self):
        """Flips a random spin in the lattice"""
        row = np.random.randint(0, high=self._lattice.shape[0])
        column = np.random.randint(0, high=self._lattice.shape[1])
        self.flip_spin(row, column)
        return row, column

    def plot_lattice(self):
        """Creates a simple matshow of the spin configuration"""
        fig, ax = plt.subplots(1)
        ax.matshow(self._lattice)
        return fig, ax


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
        SpinConfiguration(SpinConfiguration.init_random(10, 10)._lattice)  # Check if init_random returns a valid lattice
        with self.assertRaises(ValueError):
            SpinConfiguration(np.array([[1, 2], [-1, 1]]))

    def test_accessors(self):
        some_lattice = np.array([[1, -1], [-1, -1]])
        config = SpinConfiguration(some_lattice)
        self.assertEqual(config[0, 0], 1)
        self.assertEqual(config[0, 1], -1)

    def test_magnetization(self):
        some_lattice = np.array([[1, -1], [-1, -1]])
        config = SpinConfiguration(some_lattice)
        magnetization = -2
        self.assertEqual(config.magnetization(), magnetization)

    def test_energy(self):
        configuration = SpinConfiguration(np.array([[1, -1], [-1, -1]]))
        magnetic_field, coupling = 1, 1
        energy = 2  # Manual calculation
        self.assertEqual(energy, configuration.energy(magnetic_field, coupling))

    def test_flip(self):
        some_lattice = np.array([[1, -1], [-1, -1]])
        flipped = np.array([[1, 1], [-1, -1]])
        configuration = SpinConfiguration(some_lattice)
        configuration.flip_spin(0, 1)
        np.testing.assert_array_equal(configuration._lattice, flipped)
        random_flipped = SpinConfiguration(some_lattice)
        row, column = random_flipped.flip_random()
        self.assertEqual(random_flipped[row, column], some_lattice[row, column]*-1)

    def test_plot(self):
        configuration = SpinConfiguration.init_random(100, 100)
        configuration.plot_lattice()


if __name__ == '__main__':
    unittest.main()
