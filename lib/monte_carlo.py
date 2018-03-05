"""This module contains classes used for the Monte Carlo / Magnetism assignment."""
import copy
import numpy as np
import matplotlib.pyplot as plt
import lib.util as util


class MonteCarlo(object):
    """Base Monte Carlo simulator, defines virtual interface"""
    def __init__(self, num_runs):
        self._num_runs = num_runs
        self._iteration_number = 0
        self._done = False

    def init_state(self):
        """Initialize Monte Carlo simulator state"""
        pass

    def plot_state(self, *args, **kwargs):
        """Plot the Monte Carlo simulator state"""
        pass

    def is_done(self):
        """Return whether the simulation is done"""
        return self._done

    def simulate(self):
        """Perform the Monte Carlo simulation"""
        pass

    def reset(self):
        """Reset the simulation"""
        self.__init__(self._num_runs)

    """private"""
    def _iterate(self):
        """Do one Monte Carlo iteration"""
        pass


class MagnetSolver(MonteCarlo):
    """Ising model magnet Monte Carlo solver, base class for more specific models.
    Uses the Metropolis algorithm for accepting or rejecting a move.
    """
    def __init__(self, num_runs, lattice_side, unit_step=False):
        super().__init__(num_runs)
        self._num_units = num_runs
        self._lattice_side = lattice_side
        self._unit_step = unit_step
        self.energies = np.zeros(self._num_units+1)
        self.magnetizations = np.zeros(self._num_units+1)
        self.configuration = self.init_state()

    def reset(self):
        """Reset the Monte Carlo simulator state"""
        self.__init__(self._num_units, self._lattice_side)

    def set_lattice_side(self, new_side):
        """Change the lattice side"""
        self._lattice_side = new_side
        if self._unit_step:
            self._num_runs = self._num_units*self._lattice_side**2

    def init_state(self):
        """Initialize the Monte Carlo simulator state"""
        return SpinConfiguration.init_random(self._lattice_side, self._lattice_side)

    def simulate(self):
        """Run num_runs iterations and collect results"""
        assert not self._done
        if self._unit_step:
            magnetizations_buf = np.zeros(self._lattice_side**2 + 1)
            energies_buf = np.zeros(self._lattice_side**2 + 1)
            for self._unit_number in range(self._num_runs):
                magnetizations_buf[0] = self.magnetizations[self._unit_number]
                energies_buf[0] = self.energies[self._unit_number]
                for self._iteration_number in range(self._lattice_side**2):
                    magnetization_difference, energy_difference = self._iterate()
                    magnetizations_buf[self._iteration_number+1] = magnetizations_buf[self._iteration_number] \
                        + magnetization_difference
                    energies_buf[self._iteration_number+1] = energies_buf[self._iteration_number] + energy_difference
                self.magnetizations[self._unit_number+1] = magnetizations_buf[-1]
                self.energies[self._unit_number+1] = energies_buf[-1]
        else:
            for self._iteration_number in range(self._num_runs):
                magnetization_difference, energy_difference = self._iterate()
                self.magnetizations[self._iteration_number+1] = self.magnetizations[self._iteration_number] \
                    + magnetization_difference
                self.energies[self._iteration_number+1] = self.energies[self._iteration_number] + energy_difference
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

    def plot_state(self):
        """Plot the current state of the Monte Carlo simulation"""
        return self.configuration.plot_lattice()

    @staticmethod
    def move_accepted(energy_difference):
        """Return whether to accept a move"""
        if energy_difference <= 0:
            return True
        boltzmann_factor = np.exp(-energy_difference)
        random = np.random.random()
        return random < boltzmann_factor

    """private"""
    def _energy_difference(self, flipped_row, flipped_column):
        """Calculate the energy difference between two consecutive iterations"""
        return 0

    def _magnetization_difference(self, flipped_row, flipped_column):
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
        return magnetization_difference, energy_difference


class ParaMagnet(MagnetSolver):
    """Ising model paramagnet Monte Carlo solver, inherits from Magnet class."""
    def __init__(self, num_runs, magnetic_field, lattice_side, unit_step=False):
        super().__init__(num_runs, lattice_side, unit_step=unit_step)
        self._magnetic_field = magnetic_field
        self.energies[0] = self.configuration.energy(magnetic_field, 0)
        self.magnetizations[0] = self.configuration.magnetization()

    def reset(self):
        self.__init__(self._num_runs, self._magnetic_field, self._lattice_side)

    def set_magnetic_field(self, new_field):
        """Change the magnetic field"""
        self._magnetic_field = new_field

    def _energy_difference(self, flipped_row, flipped_column):
        return -2*self._magnetic_field*self.configuration[flipped_row, flipped_column]

    def _magnetization_difference(self, flipped_row, flipped_column):
        return 2*self.configuration[flipped_row, flipped_column]


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

    """private"""
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

