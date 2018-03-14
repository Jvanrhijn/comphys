"""This module contains classes needed for the Wave Propagation project"""
import numpy as np


class BaseMatrixSolver:

    def __init__(self, grid, potential, energy):
        self._grid = grid
        self._energy = energy
        self._num_factors = len(self._grid)
        self._potential = potential(grid)

    def calculate(self):
        """Calculate the matrix representing the potential barrier/well.
        return transmission and reflection coefficients
        """
        return 0

    # private
    def _product(self, first, second):
        """Multiplication rule for two matrix factors"""
        return 0

    def _matrix_factor(self, index):
        """Calculate matrix factor [index] in total product"""
        return 0


class TransferMatrixSolver(BaseMatrixSolver):

    def __init__(self, grid, potential, energy):
        super().__init__(grid, potential, energy)
        self._grid_diff = np.diff(self._grid)

    # private
    def _product(self, first, second):
        return np.dot(first, second)

    def _matrix_factor(self, index):
        pass

    def _p_submatrix(self, index):
        """Calculate matrix P(x_i)"""
        wave_vector_here = np.sqrt(self._potential[index] - self._energy) if self._energy <= self._potential[index]\
            else 1j*np.sqrt(self._energy - self._potential[index])
        wave_vector_prev = np.sqrt(self._potential[index-1] - self._energy) if self._energy <= self._potential[index-1]\
            else 1j*np.sqrt(self._energy - self._potential[index-1])
        p = 1/(2*wave_vector_prev)*np.array([[wave_vector_prev + wave_vector_here, wave_vector_prev - wave_vector_here],
                                             [wave_vector_prev - wave_vector_here, wave_vector_here + wave_vector_prev]])
        return p

    def _q_submatrix(self, index):
        wave_vector_here = np.sqrt(self._potential[index] - self._energy) if self._energy <= self._potential[index]\
            else 1j*np.sqrt(self._energy - self._potential[index])
        q = np.array([[np.exp(wave_vector_here*self._grid_diff[index]), 0],
                      [0, np.exp(-wave_vector_here*self._grid_diff[index])]])
        return q


class ScatterMatrixSolver(BaseMatrixSolver):

    def __init__(self, grid, potential, energy,):
        super().__init__(grid, potential, energy)

    # private
    def _product(self, first, second):
        return np.array([[first[0, 0] + first[0, 1]*second[0, 0]*first[1, 0]/(1 - first[1, 1]*second[0, 0]),
                          first[0, 1]*second[0, 1]/(1 - second[0, 0]*first[1, 1])],
                         [second[1, 0]*first[1, 0]/(1 - first[1, 1]*second[0, 0]),
                          second[1, 1]*second[1, 0]*first[1, 1]*second[0, 1]/(1 - first[1, 1]*second[0, 0])]])

    def _matrix_factor(self, index):
        return 0

