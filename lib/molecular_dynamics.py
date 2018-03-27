import cmath
import numpy as np
import matplotlib.pyplot as plt


class State:
    def __init__(self, num_particles, dim=3):
        self._num_particles = num_particles
        self._dim = dim
        self._num_particles = num_particles
        self._positions = np.zeros((dim, num_particles))
        self._velocities = np.zeros((dim, num_particles))

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities

    def forces(self, forces_function) -> np.ndarray:
        return forces_function(self._positions)

    def init_random(self, position_range: tuple, velocity_range: tuple):
        self._positions = np.random.random(size=(self._dim, self._num_particles))\
            * (position_range[1] - position_range[0]) + position_range[0]
        self._velocities = np.random.random(size=(self._dim, self._num_particles))\
            * (velocity_range[1] - velocity_range[0]) + velocity_range[0]
        return self

