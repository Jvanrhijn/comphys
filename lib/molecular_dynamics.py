"""Classes for use in the Molecular Dynamics project"""
import cmath
import numpy as np
import matplotlib.pyplot as plt


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

    @positions.setter
    def positions(self, new_pos) -> None:
        self._positions = new_pos

    @velocities.setter
    def velocities(self, new_vel) -> None:
        self._velocities = new_vel

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


class MDSimulator:
    """Molecular dynamics simulator class"""
    def __init__(self, init_state, integrator, time_step, num_steps, force_function):
        self._integrator = integrator(init_state, force_function, time_step, max_steps=num_steps)
        self._forces = force_function
        self._time_step = time_step
        self._num_steps = num_steps
        self._end_time = num_steps*time_step
        self._state_vars = {}
        self._state_functions = []
        self._step = 0

    def simulate(self):
        """Perform the molecular dynamics simulation with the given parameters"""
        for self._step, state in enumerate(self._integrator):
            self._calc_state_vars()
        return self._integrator.state

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
    def state_vars(self):
        """State variable getter"""
        return self._state_vars

    # Private
    def _calc_state_vars(self):
        for state_func in self._state_functions:
            self._state_vars[state_func[0]][self._step] = state_func[1](self._integrator.state)
