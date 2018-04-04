"""Classes for use in the Molecular Dynamics project"""
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from tqdm import tqdm


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

    def kinetic_energy(self) -> float:
        return 0.5*np.sum(self._velocities**2)

    def temperature(self) -> float:
        return 2*self.kinetic_energy()/(self._num_particles*self._dim)

    def set_temperature(self, new_temp) -> None:
        scale_factor = np.sqrt(new_temp/self.temperature())
        self.velocities *= scale_factor

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

    def init_grid(self, box):
        offset = 0.5
        p_per_side = math.ceil(self._num_particles**(1/3))
        x, y, z = np.meshgrid(np.linspace(offset, box.side(0)-offset, p_per_side),
                              np.linspace(offset, box.side(1)-offset, p_per_side),
                              np.linspace(offset, box.side(2)-offset, p_per_side))
        for p in range(self._num_particles):
            self._positions[:, p] = np.array([x.flatten()[p], y.flatten()[p], z.flatten()[p]])
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

    @property
    def state(self):
        """Current state getter"""
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state


class Simulator:
    """Molecular dynamics simulator class"""
    def __init__(self, init_state, integrator, time_step, num_steps, force_function):
        self._integrator = integrator(init_state, force_function, time_step, max_steps=num_steps)
        self._init_integrator = copy.deepcopy(integrator)
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
        return self._integrator.state

    def simulate(self):
        """Perform the molecular dynamics simulation with the given parameters"""
        if self.save:
            for self._step, state in enumerate(tqdm(self._integrator, total=self._num_steps)):
                self._calc_state_vars()
                self._states[self._step] = copy.deepcopy(state)
            return self._integrator.state
        for self._step, state in enumerate(tqdm(self._integrator, total=self._num_steps)):
            self._calc_state_vars()
        return self._integrator.state

    def advance_state(self):
        """Advance to the next simulator state"""
        next(self._integrator)
        self._calc_state_vars()
        if self.save:
            self._states[self._step] = copy.deepcopy(self.state())

    def set_state_vars(self, *args) -> None:
        """
        Set the state variables to calculate at each time step
        :param args: Tuples of names and the functions used to calculate the state variables

        Example usage:
        simulator.set_state_vars(
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
            self._state_vars[state_func[0]][self._step] = state_func[1](self._integrator.state)


class Box:
    def __init__(self, *sides):
        self._sides = sides

    @property
    def sides(self):
        return self._sides

    @property
    def volume(self):
        volume = 1
        for side_length in self._sides:
            volume *= side_length
        return volume

    def side(self, index):
        return self._sides[index]


class BoxedSimulator(Simulator):

    def __init__(self, init_state, integrator, time_step, num_steps, force_func, box):
        super().__init__(init_state, integrator, time_step, num_steps, force_func)
        self._box = box
        self._constraints = np.array(list(box.sides))

    def simulate(self):
        """Perform the molecular dynamics simulation with the given parameters"""
        if self.save:
            for self._step, state in enumerate(tqdm(self._integrator, total=self._num_steps)):
                self._apply_constraints()
                self._calc_state_vars()
                self._states[self._step] = copy.deepcopy(state)
            return self._integrator.state
        for self._step, state in enumerate(tqdm(self._integrator, total=self._num_steps)):
            self._apply_constraints()
            self._calc_state_vars()
        return self._integrator.state

    def advance_state(self):
        """Advance to the next simulator state"""
        next(self._integrator)
        self._apply_constraints()
        self._calc_state_vars()
        if self.save:
            self._calc_state_vars()
            self._states[self._step] = copy.deepcopy(self.state())

    # Private
    def _apply_constraints(self):
        for dim in range(self._integrator.state.dim):
            self._integrator.state.positions[dim, :] %= self._constraints[dim]


class BoxedNVESimulator(BoxedSimulator):

    def __init__(self, init_state, integrator, time_step, num_steps, force_func, box, temp_target, prep_steps):
        super().__init__(init_state, integrator, time_step, num_steps, force_func, box)
        self._temp_target = temp_target
        self._prep_steps = prep_steps

    def simulate(self):
        """Perform the molecular dynamics simulation with the given parameters"""
        if self.save:
            for self._step in tqdm(range(self._prep_steps)):
                next(self._integrator)
                self._integrator.state.set_temperature(self._temp_target)
                self._apply_constraints()
                self._calc_state_vars()
                self._states[self._step] = copy.deepcopy(self._integrator.state)
            for self._step in tqdm(range(self._prep_steps, self._num_steps)):
                next(self._integrator)
                self._apply_constraints()
                self._calc_state_vars()
                self._states[self._step] = copy.deepcopy(self._integrator.state)
            return self._integrator.state
        for self._step in tqdm(range(self._prep_steps)):
            next(self._integrator)
            self._integrator.state.set_temperature(self._temp_target)
            self._apply_constraints()
            self._calc_state_vars()
        for self._step in tqdm(range(self._prep_steps, self._num_steps)):
            next(self._integrator)
            self._apply_constraints()
            self._calc_state_vars()
        return self._integrator.state


class Visualizer:
    """Class that provides various visualizations for a running simulation"""
    def __init__(self, simulator, inf_sim=False):
        self._simulator = simulator
        self._points = None
        self._ax = None
        if inf_sim:
            if self._simulator.save:
                raise ValueError("Cannot save states if continuing simulation indefinitely")
            simulator._integrator._max_steps = np.inf

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

    def particle_cloud_animation(self, num_frames, interval, *args, xaxis_bounds=None, yaxis_bounds=None,
                                 zaxis_bounds=None, **kwargs):
        fig = plt.figure()
        ax = Axes3D(fig)
        if xaxis_bounds:
            ax.set_xlim(xaxis_bounds[0], xaxis_bounds[1])
        if yaxis_bounds:
            ax.set_ylim(yaxis_bounds[0], yaxis_bounds[1])
        if zaxis_bounds:
            ax.set_zlim(zaxis_bounds[0], zaxis_bounds[1])
        self._ax = ax
        xs = self._simulator.state().positions[0, :]
        ys = self._simulator.state().positions[1, :]
        zs = self._simulator.state().positions[2, :]
        # initial scatter plot
        self._points, = ax.plot(xs, ys, zs, 'o', *args, **kwargs)

        anim = animation.FuncAnimation(fig, self._update_cloud, num_frames,
                                       interval=interval, repeat=True, blit=True)
        return fig, ax, anim

    # Private
    def _update_cloud(self, i):
        assert self._points is not None
        self._simulator.advance_state()
        xs = self._simulator.state().positions[0, :]
        ys = self._simulator.state().positions[1, :]
        zs = self._simulator.state().positions[2, :]
        self._points.set_data(np.array([xs, ys]))
        self._points.set_3d_properties(zs, 'z')
        return self._points,

