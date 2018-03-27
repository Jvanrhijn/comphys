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
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities

    @positions.setter
    def positions(self, new_pos) -> None:
        self._positions = new_pos

    @velocities.setter
    def velocities(self, new_vel) -> None:
        self._velocities = new_vel

    def forces(self, forces_function) -> np.ndarray:
        """Calculate the forces on the current state given a force function"""
        return forces_function(self)

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


class Integrator:
    """Base integrator class"""
    def __init__(self, init_state, force_function, time_step):
        self._state = init_state
        self._forces = force_function
        self._time_step = time_step


class VerletIntegrator(Integrator):
    """Integrator/iterator that implements Verlet algorithm"""
    def __iter__(self):
        self._half_velocity = 0.
        return self

    def __next__(self):
        self._half_velocity = self._state.velocities + 0.5*self._forces(self._state)*self._time_step
        self._state.positions += self._half_velocity*self._time_step
        self._state.velocities = self._half_velocity + 0.5*self._forces(self._state)*self._time_step
        return self._state

