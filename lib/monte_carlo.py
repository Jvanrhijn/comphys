import numpy as np
import copy
import matplotlib.pyplot as plt
import lib.util as util
import unittest


class MonteCarlo(object):
    """Base Monte Carlo simulator, defines virtual interface"""
    def __init__(self, num_runs):
        self._num_runs = num_runs
        self._iteration_number = 0
        self._done = False

    def init_state(self):
        pass

    def _iterate(self):
        pass

    def plot_state(self, *args, **kwargs):
        pass

    def variable_error(self):
        pass

    def is_done(self):
        return self._done

    def reset(self):
        self.__init__(self._num_runs)


class MagnetSolver(MonteCarlo):
    """Ising model magnet Monte Carlo solver, base class for more specific models.
    Uses the Metropolis algorithm for accepting or rejecting a move.
    """
    def __init__(self, num_runs, lattice_side):
        super().__init__(num_runs)
        self._lattice_side = lattice_side
        self.energies = np.zeros(num_runs+1)
        self.magnetizations = np.zeros(num_runs+1)
        self.configuration = self.init_state()

    def reset(self):
        """Reset the Monte Carlo simulator state"""
        self.__init__(self._num_runs, self._lattice_side)

    def set_lattice_side(self, new_side):
        self._lattice_side = new_side

    def init_state(self):
        """Initialize the Monte Carlo simulator state"""
        return SpinConfiguration.init_random(self._lattice_side, self._lattice_side)

    @staticmethod
    def move_accepted(energy_difference):
        """Return whether to accept a move"""
        if energy_difference <= 0:
            return True
        else:
            boltzmann_factor = np.exp(-energy_difference)
            random = np.random.random()
            return random < boltzmann_factor

    def _energy_difference(self, *args):
        """Calculate the energy difference between two consecutive iterations"""
        return 0

    def _magnetization_difference(self, *args):
        """Calculate the magnetization difference between two consecutive iterations"""
        return 0

    def _iterate(self):
        """Do one Monte Carlo iteration, and save energy and magnetization"""
        row, column = self.configuration.flip_random()
        energy_difference = self._energy_difference(row, column)
        magnetization_difference = self._magnetization_difference(row, column)
        if not self.move_accepted(energy_difference):
            self.configuration.flip_spin(row, column)  # Restore old configuration
            energy_difference = 0
            magnetization_difference = 0
        self.energies[self._iteration_number+1] = self.energies[self._iteration_number] + energy_difference
        self.magnetizations[self._iteration_number+1] = self.magnetizations[self._iteration_number] \
            + magnetization_difference

    def simulate(self):
        """Run num_runs iterations and collect results"""
        for self._iteration_number in range(self._num_runs):
            self._iterate()
        self._done = True

    def mean_energy(self, equilibration_time):
        """Get mean energy and standard deviation"""
        assert self._done
        mean = np.mean(self.energies[equilibration_time:])
        stdev = np.std(self.energies[equilibration_time:])
        return mean, stdev

    def mean_magnetization(self, equilibration_time):
        """Get mean magnetization and standard deviation"""
        assert self._done
        mean = np.mean(self.magnetizations[equilibration_time:])/self._lattice_side**2
        stdev = np.std(self.magnetizations[equilibration_time:])/self._lattice_side**2
        return mean, stdev


class ParaMagnet(MagnetSolver):
    """Ising model paramagnet Monte Carlo solver, inherits from Magnet class."""
    def __init__(self, num_runs, magnetic_field, lattice_side):
        super().__init__(num_runs, lattice_side)
        self._magnetic_field = magnetic_field
        self.energies[0] = self.configuration.energy(magnetic_field, 0)
        self.magnetizations[0] = self.configuration.magnetization()

    def reset(self):
        self.__init__(self._num_runs, self._magnetic_field, self._lattice_side)

    def set_magnetic_field(self, new_field):
        self._magnetic_field = new_field

    def _energy_difference(self, flipped_row, flipped_column):
        return -2*self._magnetic_field*self.configuration[flipped_row, flipped_column]

    def _magnetization_difference(self, flipped_row, flipped_column):
        return 2*self.configuration[flipped_row, flipped_column]

    def plot_state(self):
        """Plot the current state of the Monte Carlo simulation"""
        return self.configuration.plot_lattice()


class SpinConfiguration(object):
    """Object representing spins on a lattice"""
    def __init__(self, lattice):
        if np.logical_or(np.equal(np.ones(lattice.shape), lattice), np.equal(np.ones(lattice.shape)*-1, lattice)).all():
            self._lattice = copy.deepcopy(lattice)
            self._rows = lattice.shape[0]
            self._columns = lattice.shape[1]
        else:
            raise ValueError("Spins must be either up (1) or down (-1)")

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
        magnetic_energy = 0 if magnetic_field == 0 else -magnetic_field*self.magnetization()
        exchange_energy = 0 if coupling == 0 else self._exchange_energy(coupling)
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
                exchange_energy += self._lattice[i, j]*interaction
        return -coupling*exchange_energy

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
        ax.pcolormesh(self._lattice, cmap='Greys')
        plt.axis('equal')
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

    def test_paramagnet(self):
        field = 1
        mc_paramagnet = ParaMagnet(2000, field, 10)
        exact_magnetization = np.tanh(field)
        mc_paramagnet.simulate()
        mean_magnetization, stdev = mc_paramagnet.mean_magnetization(200)
        # Test may fail in 5% of cases
        self.assertTrue(exact_magnetization - 2*stdev < mean_magnetization < exact_magnetization + 2*stdev)

    def test_reset(self):
        mc_paramagnet = ParaMagnet(2000, 1, 10)
        mc_paramagnet.simulate()
        magnetization_first = mc_paramagnet.mean_magnetization(200)[0]
        mc_paramagnet.reset()
        self.assertFalse(mc_paramagnet.is_done())
        # If reset worked, energies & magnetizations should both be zero and equal
        np.testing.assert_array_equal(mc_paramagnet.magnetizations[1:], mc_paramagnet.energies[1:])
        # After second simulation, new magnetization should be close to but not equal to previous simulation
        mc_paramagnet.simulate()
        magnetization_second = mc_paramagnet.mean_magnetization(200)[0]
        self.assertAlmostEqual(magnetization_first, magnetization_second, places=1)
        self.assertNotEqual(magnetization_first, magnetization_second)

    def test_set_functions(self):
        field = 1
        side = 10
        mc_paramagnet = ParaMagnet(2000, field, side)
        mc_paramagnet.set_magnetic_field(2)
        mc_paramagnet.set_lattice_side(11)
        self.assertEqual(11, mc_paramagnet._lattice_side)
        self.assertEqual(2, mc_paramagnet._magnetic_field)


if __name__ == '__main__':
    unittest.main()
