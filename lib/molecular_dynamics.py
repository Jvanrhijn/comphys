"""Classes for use in the Molecular Dynamics project"""
import copy
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


class State:
    """Encodes a many-particle state.
    This class encodes the state of a collection of N classical particles, in d spatial dimensions (default 3)
    """
    def __init__(self, num_particles, dim=3):
        self._num_particles = num_particles
        self._dim = dim
        self._num_particles = num_particles
        self._positions = np.zeros((dim, num_particles), dtype=float)
        self._velocities = np.zeros((dim, num_particles), dtype=float)

    @property
    def positions(self) -> np.ndarray:
        """State positions getter"""
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        """State positions getter"""
        return self._velocities

    @property
    def dim(self):
        return self._dim

    @positions.setter
    def positions(self, new_pos) -> None:
        """State positions setter"""
        if new_pos.shape != self._positions.shape:
            raise ValueError("New positions must be of shape " + str(self._positions.shape))
        self._positions = new_pos

    @velocities.setter
    def velocities(self, new_vel) -> None:
        """State velocities setter"""
        if new_vel.shape != self._velocities.shape:
            raise ValueError("New velocities must be of shape " + str(self._velocities.shape))
        self._velocities = new_vel

    def get_single_particle(self, num):
        state = State(1, dim=self._dim)
        state.positions = np.reshape(self.positions[:, num], (self._dim, 1))
        state.velocities = np.reshape(self.velocities[:, num], (self._dim, 1))
        return state

    def init_random(self, position_range: tuple, velocity_range: tuple):
        """
        Initialize the State with random positions and velocities in a given range
        :param position_range: range of positions (2-tuple)
        :param velocity_range: range of velocities (2-tuple)
        :return: Reference to the current state
        """
        self._positions = np.random.random(size=(self._dim, self._num_particles))\
            * (position_range[1] - position_range[0]) + position_range[0]
        self._velocities = np.random.random(size=(self._dim, self._num_particles))\
            * (velocity_range[1] - velocity_range[0]) + velocity_range[0]
        return self

    def center_of_mass(self) -> tuple:
        """Calculate the center of mass of the collection of particles and its velocity vector"""
        position_com = np.reshape(np.sum(self.positions, axis=1), (self._dim, 1))/self._num_particles
        velocity_com = np.reshape(np.sum(self.velocities, axis=1), (self._dim, 1))/self._num_particles
        return position_com, velocity_com


class VerletIntegrator:
    """Integrator/iterator that implements Verlet algorithm"""
    def __init__(self, init_state, force_function, time_step, max_steps=np.inf):
        self._state = init_state
        self._forces = force_function
        self._time_step = time_step
        self._max_steps = max_steps
        self._step = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._step >= self._max_steps:
            raise StopIteration
        half_velocity = self._state.velocities + 0.5*self._forces(self._state)*self._time_step
        self._state.positions += half_velocity*self._time_step
        self._state.velocities = half_velocity + 0.5*self._forces(self._state)*self._time_step
        self._step += 1
        return self._state

    def state(self):
        """Current state getter"""
        return self._state


class Simulator:
    """Molecular dynamics simulator class"""
    def __init__(self, init_state, integrator, time_step, num_steps, force_function):
        self._integrator = integrator(init_state, force_function, time_step, max_steps=num_steps)
        self._num_steps = num_steps
        self._end_time = num_steps*time_step
        self._state_vars = {}
        self._state_functions = []
        self._step = 0
        self._states = np.zeros(num_steps, dtype=State)
        self._init_state = copy.deepcopy(self.state())
        self.save = False

    def state(self):
        """Get the current state of the internal integrator"""
        return self._integrator.state()

    def simulate(self):
        """Perform the molecular dynamics simulation with the given parameters"""
        if self.save:
            for self._step, state in enumerate(self._integrator):
                self._calc_state_vars()
                self._states[self._step] = copy.deepcopy(state)
            return self._integrator.state
        for self._step, state in enumerate(self._integrator):
            self._calc_state_vars()
        return self._integrator.state()

    def advance_state(self):
        """Advance to the next simulator state"""
        if self.save:
            next(self._integrator)
            self._calc_state_vars()
            self._states[self._step] = copy.deepcopy(self.state())
        else:
            next(self._integrator)
            self._calc_state_vars()

    def reset(self):
        self._integrator.state = copy.deepcopy(self._init_state)
        for key in self._state_vars:
            self._state_vars[key] = np.zeros(self._num_steps)
        self._states = np.zeros(self._num_steps, dtype=State)

    def set_state_vars(self, *args) -> None:
        """
        Set the state variables to calculate at each time step
        :param args: Tuples of names and the functions used to calculate the state variables

        Example usage:
        simulator.state_vars(
                            ("energy", lambda s: 0.5*(np.sum(s.velocities**2) + sum(s.positions**2)),
                            ("pressure", lambda s: ...)
                            )
        """
        for arg in args:
            self._state_vars[arg[0]] = np.zeros(self._num_steps)
            self._state_functions.append((arg[0], arg[1]))

    @property
    def state_vars(self) -> dict:
        """State variable getter"""
        return self._state_vars

    @property
    def states(self) -> np.ndarray:
        """Saved states getter"""
        return self._states

    # Private
    def _calc_state_vars(self) -> None:
        for state_func in self._state_functions:
            self._state_vars[state_func[0]][self._step] = state_func[1](self._integrator.state())


class Visualizer:
    """Class that provides various visualizations for a running simulation"""
    def __init__(self, simulator):
        self._simulator = simulator
        self._points = None

    @staticmethod
    def plot_particle_cloud(state, *args):
        if state.dim != 3:
            raise ValueError("Cloud must be three-dimensional")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = state.positions[0, :]
        ys = state.positions[1, :]
        zs = state.positions[2, :]
        ax.scatter(xs, ys, zs, *args)
        return fig, ax

    def particle_cloud_animation(self, num_frames, interval, *args):
        fig = plt.figure()
        ax = Axes3D(fig)
        xs = self._simulator.state().positions[0, :]
        ys = self._simulator.state().positions[1, :]
        zs = self._simulator.state().positions[2, :]
        # initial scatter plot
        self._points, = ax.plot(xs, ys, zs, 'o')

        anim = animation.FuncAnimation(fig, self._update_cloud, frames=num_frames,
                                       interval=interval, repeat=True)
        return fig, ax, anim

    def _update_cloud(self, i):
        assert self._points is not None
        try:
            self._simulator.advance_state()
        except StopIteration:
            self._simulator.state().init_random(self._simulator)
        xs = self._simulator.state().positions[0, :]
        ys = self._simulator.state().positions[1, :]
        zs = self._simulator.state().positions[2, :]
        self._points.set_data(np.array([xs, ys]))
        self._points.set_3d_properties(zs, 'z')
        return self._points

