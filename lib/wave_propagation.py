"""This module contains classes needed for the Wave Propagation project"""
import numpy as np


class BaseMatrixSolver:

    def __init__(self, grid, potential, energy):
        self._grid = grid
        self._energy = energy
        self._num_factors = len(self._grid)
        self._potential = potential(grid)

    def solve(self):
        """Calculate the matrix representing the potential barrier/well.
        return transmission and reflection coefficients
        """
        pass

    # private
    def _product(self, first, second):
        """Multiplication rule for two matrix factors"""
        pass

    def _matrix_factor(self, index):
        """Calculate matrix factor [index] in total product"""
        pass


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
        pass

    def _q_submatrix(self, index):
        pass


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
        pass

