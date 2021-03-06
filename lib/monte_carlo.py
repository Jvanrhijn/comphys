"""This module contains classes used for the Monte Carlo / Magnetism assignment."""
import copy
import numpy as np
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda *i, **kwargs: i[0]
from matplotlib.gridspec import GridSpec
from matplotlib import rc
import lib.util as util
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 12})
rc('text', usetex=True)


class MonteCarlo(object):
    """Base Monte Carlo simulator, defines virtual interface"""
    def __init__(self, num_runs):
        self._num_runs = num_runs
        self._iteration_number = 0
        self._equilibration_time = 0
        self._done = False

    def init_state(self):
        """Initialize Monte Carlo simulator state"""
        pass

    @property
    def equilibration_time(self):
        return self._equilibration_time

    @equilibration_time.setter
    def equilibration_time(self, equilibration_time):
        """Set the equilibration time for the simulation"""
        self._equilibration_time = equilibration_time

    @property
    def num_runs(self):
        return self._num_runs

    @num_runs.setter
    def num_runs(self, new_num_runs):
        """Accessor method for setting the number of runs"""
        self._num_runs = new_num_runs
        self.reset()

    def plot_results(self):
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

    # Private
    def _iterate(self):
        """Do one Monte Carlo iteration"""
        pass


class MagnetSolver(MonteCarlo):
    """Ising model magnet Monte Carlo solver, base class for more specific models.
    Uses the Metropolis algorithm for accepting or rejecting a move.
    """
    def __init__(self, num_runs, lattice_side):
        super().__init__(num_runs)
        self._unit_number = 0
        self._lattice_side = lattice_side
        self.energies = np.zeros(self._num_runs+1)
        self.magnetizations = np.zeros(self._num_runs+1)
        self.correlations = np.zeros((self._num_runs+1, self._lattice_side//2))
        self.configuration = self.init_state()
        self.correlations[0, :] = self.configuration.correlation()

    def reset(self):
        """Reset the Monte Carlo simulator state"""
        self.__init__(self._num_runs, self._lattice_side)

    @property
    def lattice_side(self):
        return self._lattice_side

    @lattice_side.setter
    def lattice_side(self, new_side):
        """Change the lattice side"""
        self._lattice_side = new_side
        self.reset()

    def init_state(self):
        """Initialize the Monte Carlo simulator state"""
        return SpinConfiguration.init_random(self._lattice_side, self._lattice_side)

    def simulate_unit(self, pbar=True):
        """Run num_runs unit sweeps of lattice_side**2 spin flips"""
        assert not self._done
        # Buffers for magnetization and energy in each unit step
        magnetizations_buf = np.zeros(self._lattice_side**2 + 1)
        energies_buf = np.zeros(self._lattice_side**2 + 1)
        if pbar:
            iterator = tqdm(range(self._num_runs))
        else:
            iterator = range(self._num_runs)
        for self._unit_number in iterator:
            # Initialize first value of buffer to value of previous unit step
            magnetizations_buf[0] = self.magnetizations[self._unit_number]
            energies_buf[0] = self.energies[self._unit_number]
            # Do the unit step
            for self._iteration_number in range(self._lattice_side**2):
                magnetization_difference, energy_difference = self._iterate()
                magnetizations_buf[self._iteration_number+1] = magnetizations_buf[self._iteration_number] \
                                                               + magnetization_difference
                energies_buf[self._iteration_number+1] = energies_buf[self._iteration_number] + energy_difference
            # Save unit step results
            self.magnetizations[self._unit_number+1] = magnetizations_buf[-1]
            self.energies[self._unit_number+1] = energies_buf[-1]
            self.correlations[self._unit_number+1] = self.configuration.correlation()
        self._done = True

    def simulate(self, pbar=True):
        """Run num_runs iterations and collect results"""
        assert not self._done
        if pbar:
            iterator = tqdm(range(self._num_runs))
        else:
            iterator = range(self._num_runs)
        # If not using unit steps, just do the iteration normally, saving values for each spin flip
        for self._iteration_number in iterator:
            magnetization_difference, energy_difference = self._iterate()
            self.magnetizations[self._iteration_number+1] = self.magnetizations[self._iteration_number] \
                                                            + magnetization_difference
            self.energies[self._iteration_number+1] = self.energies[self._iteration_number] + energy_difference
            self.correlations[self._unit_number+1] = self.configuration.correlation()
        self._done = True

    def mean_energy(self):
        """Get mean energy per site and standard deviation"""
        assert self._done
        mean = np.mean(self.energies[self._equilibration_time:])/self._lattice_side**2
        error = np.std(self.energies[self._equilibration_time:])/self._lattice_side**2/np.sqrt(self._num_runs-1)
        return mean, error

    def mean_magnetization(self, absolute=False):
        """Get mean magnetization per site and standard deviation"""
        assert self._done
        if not absolute:
            mean = np.mean(self.magnetizations[self._equilibration_time:])/self._lattice_side**2
            error = np.std(self.magnetizations[self._equilibration_time:])/self._lattice_side**2/(np.sqrt(self._num_runs-1))
        else:
            mean = np.mean(np.abs(self.magnetizations[self._equilibration_time:]))/self._lattice_side**2
            error = np.std(np.abs(self.magnetizations[self._equilibration_time:]))\
                / self._lattice_side**2/(np.sqrt(self._num_runs-1))
        return mean, error

    def susceptibility(self, absolute=False):
        """Get magnetic susceptibility per site"""
        assert self._done
        if not absolute:
            return (np.mean(self.magnetizations[self._equilibration_time:]**2)
                    - np.mean(self.magnetizations[self._equilibration_time:])**2)/self._lattice_side**2
        else:
            return (np.mean(abs(self.magnetizations[self._equilibration_time:])**2)
                    - np.mean(abs(self.magnetizations[self._equilibration_time:]))**2)/self._lattice_side**2

    def correlation(self):
        """Return the spin-spin correlation function"""
        assert self._done
        return np.mean(self.correlations[self.equilibration_time:], 0)/self._lattice_side**2 \
            - self.mean_magnetization()[0]**2

    def heat_capacity(self):
        """Return the heat capacity per site of the magnet at constant magnetic field"""
        return np.mean(self.energies[self._equilibration_time:]**2)/self._lattice_side**2 - self.mean_energy()[0]

    def plot_correlation(self, ax, *args, **kwargs):
        """Plot the spin-spin correlation function"""
        correlation = self.correlation()
        distance = np.linspace(1, self._lattice_side//2, len(correlation))
        ax.plot(distance, correlation, *args, **kwargs)
        return ax

    def plot_results(self):
        """Plot the current state of the Monte Carlo simulation"""
        fig = plt.figure()
        grid_spec = GridSpec(2, 2)
        ax_magnetization = plt.subplot(grid_spec[0, 1])
        ax_energies = plt.subplot(grid_spec[1, 1])
        ax_lattice = plt.subplot(grid_spec[0, 0])
        ax_text = plt.subplot(grid_spec[1, 0])
        ax_magnetization.plot(self.magnetizations/self._lattice_side**2)
        ax_magnetization.set_ylabel(r"$m(S)$")
        ax_energies.plot(self.energies/self._lattice_side**2, 'g')
        ax_energies.set_ylabel(r"$E(S)/N$")
        magnetization, m_stdev = self.mean_magnetization()
        energy, e_stdev = self.mean_energy()
        text = r'$N = L^2 = {}$'.format(self._lattice_side**2) + "\n" +\
            r'$\kappa = {}$'.format(self._equilibration_time) + "\n" +\
            r'$N_{{MC}} = {}$'.format(self._num_runs) + "\n" +\
            r'------' + "\n" +\
            r'$\langle m \rangle = {0} \pm {1}$'.format(round(magnetization, 4), round(m_stdev, 4)) + "\n" +\
            r'$\langle E/N \rangle = {0} \pm {1}$'.format(round(energy, 4), round(e_stdev, 4)) + "\n" +\
            r'$\chi = {}$'.format(round(self.susceptibility(), 4))
        ax_text.text(0, 0.3, text, fontsize=11)
        ax_text.axis('off')
        self.configuration.plot_lattice(ax_lattice)
        ax_magnetization.grid(True)
        ax_energies.grid(True)
        axes = [ax_lattice, ax_magnetization, ax_energies, ax_text]
        grid_spec.tight_layout(fig, h_pad=0, w_pad=0)
        return fig, axes

    @staticmethod
    def move_accepted(energy_difference):
        """Return whether to accept a move"""
        if energy_difference <= 0:
            return True
        boltzmann_factor = np.exp(-energy_difference)
        random = np.random.random()
        return random < boltzmann_factor

    # Private
    def _energy_difference(self, flipped_row, flipped_column):
        """Calculate the energy difference between two consecutive iterations"""
        pass

    def _magnetization_difference(self, flipped_row, flipped_column):
        """Calculate the magnetization difference between two consecutive iterations"""
        return 2*self.configuration[flipped_row, flipped_column]

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
    def __init__(self, num_runs, magnetic_field, lattice_side):
        super().__init__(num_runs, lattice_side)
        self._magnetic_field = magnetic_field
        self.energies[0] = self.configuration.energy(magnetic_field, 0)
        self.magnetizations[0] = self.configuration.magnetization()

    def reset(self):
        self.__init__(self._num_runs, self._magnetic_field, self._lattice_side)

    @property
    def magnetic_field(self):
        return self._magnetic_field

    @magnetic_field.setter
    def magnetic_field(self, new_field):
        """Change the magnetic field"""
        self._magnetic_field = new_field
        self.reset()

    def _energy_difference(self, flipped_row, flipped_column):
        return -2*self._magnetic_field*self.configuration[flipped_row, flipped_column]


class FerroMagnet(MagnetSolver):
    """Ising model ferromagnet solver, inherits from magnet class"""
    def __init__(self, num_runs, magnetic_field, coupling, lattice_side):
        super().__init__(num_runs, lattice_side)
        self._magnetic_field = magnetic_field
        self._coupling = coupling
        self.energies[0] = self.configuration.energy(magnetic_field, coupling)
        self.magnetizations[0] = self.configuration.magnetization()

    def reset(self):
        self.__init__(self._num_runs, self._magnetic_field, self._coupling, self._lattice_side)

    @property
    def coupling(self):
        return self._coupling

    @coupling.setter
    def coupling(self, new_coupling):
        """Accessor method for ferromagnetic coupling"""
        self._coupling = new_coupling
        self.reset()

    @property
    def magnetic_field(self):
        return self._magnetic_field

    @magnetic_field.setter
    def magnetic_field(self, new_field):
        """Accessor method for magnetic field"""
        self._magnetic_field = new_field
        self.reset()

    def _energy_difference(self, flipped_row, flipped_column):
        flipped_spin = self.configuration[flipped_row, flipped_column]
        magnetic_energy_difference = -2*self._magnetic_field*flipped_spin
        exchange_energy_difference = -2*self._coupling*flipped_spin*(
            self.configuration[flipped_row-1, flipped_column]
            + self.configuration[(flipped_row+1) % self._lattice_side, flipped_column]
            + self.configuration[flipped_row, flipped_column-1]
            + self.configuration[flipped_row, (flipped_column + 1) % self._lattice_side])
        return magnetic_energy_difference + exchange_energy_difference


class SpinConfiguration(object):
    """Object representing spins on a lattice"""
    def __init__(self, lattice):
        if lattice.shape == (0, 0):
            raise ValueError("Cannot construct emppty lattice")
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

    def __setitem__(self, key, value):
        """Set a spin to a value"""
        assert(value in [-1, 1])
        self._lattice[key] = value

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

    def correlation(self):
        """Returns <s_k*s_{k+r}>_k*rows*columns"""
        total = 0
        for column in range(self._rows):
            correlating_spins = np.concatenate((self._lattice[:, column+1:],
                                                self._lattice[:, :column+1]), axis=1)[:, :self._columns//2]
            total += np.dot(self._lattice[:, column], correlating_spins)
        return total

    def flip_spin(self, row, column):
        """Flip a given spin in the lattice"""
        self._lattice[row, column] *= -1

    def flip_random(self):
        """Flips a random spin in the lattice"""
        row = np.random.randint(0, high=self._lattice.shape[0])
        column = np.random.randint(0, high=self._lattice.shape[1])
        self.flip_spin(row, column)
        return row, column

    def plot_lattice(self, ax):
        """Creates a simple matshow of the spin configuration"""
        cmap = plt.cm.get_cmap('Greys', 2)
        colormesh = ax.pcolormesh(self._lattice, cmap=cmap)
        ax.axis('equal')
        ax.tick_params(
            which='both',
            bottom='off',
            top='off',
            left='off',
            right='off',
            labelbottom='off',
            labelleft='off'
        )
        ax.set_title('Lattice')
        plt.colorbar(colormesh, ax=ax, ticks=[-1, 1])
        return ax

    # Private
    def _exchange_energy(self, coupling):
        """Return exchange energy of the lattice"""
        exchange_energy = 0
        for i in range(0, self._rows):
            for j in range(0, self._columns):
                # Interaction energy with periodic BC, don't need to modulo for negative indices since in Python
                # list[-1] == list[len(list)-1].
                interaction = self._lattice[i-1, j]\
                              + self._lattice[(i+1) % self._rows, j]\
                              + self._lattice[i, j-1]\
                              + self._lattice[i, (j+1) % self._columns]
                exchange_energy += self._lattice[i, j]*interaction
        return -0.5*coupling*exchange_energy
