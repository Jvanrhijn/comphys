"""This module contains classes needed for the Wave Propagation project"""
import numpy as np


class BaseMatrix:

    def __init__(self, value=None):
        """Base quantum matrix class"""
        if value is not None:
            if value.shape != (2, 2):
                raise ValueError("Matrix dimension must be 2x2")
            self._value = value
        else:
            self._value = np.zeros((2, 2))

    def __get__(self, key):
        return self._value[key]

    def __set__(self, key, value):
        self._value[key] = value

    def transmission(self):
        """Get transmission coefficients of unit wave incoming from left and right, respectively"""
        return 0, 0

    def reflection(self):
        """Get reflection coefficients of unit wave incoming from left and right, respectively"""
        return 0, 0


class BaseMatrixSolver:

    def __init__(self, grid, potential, energy):
        self._grid = grid
        self._energy = energy
        self._num_factors = len(self._grid)
        self._potential = potential(grid)
        self._matrix = BaseMatrix

    def calculate(self):
        """Calculate the matrix representing the potential barrier/well.
        return
        """
        total_product = np.identity(2)
        for index in range(1, self._num_factors):
            total_product = self._product(total_product, self._matrix_factor(index))
        return self._matrix(value=total_product)

    def _wave_vector(self, index):
        """Calculate the local wave vector (\eta_i in the lecture notes)"""
        potential = self._potential[index]
        if self._energy <= potential:
            wave_vector = np.sqrt(potential - self._energy)
        else:
            wave_vector = np.sqrt(self._energy - potential)*1j
        return wave_vector

    def _product(self, first, second):
        """Multiplication rule for two matrix factors"""
        return 0

    def _matrix_factor(self, index):
        """Calculate matrix factor [index] in total product"""
        return 0


class TransferMatrixSolver(BaseMatrixSolver):

    def __init__(self, grid, potential, energy):
        super().__init__(grid, potential, energy)
        self._matrix = TransferMatrix

    # private
    def _product(self, first, second):
        return first @ second

    def _matrix_factor(self, index):
        """Calculate factor M(x_i) product of transfer matrices"""
        wave_vector_prev = self._wave_vector(index-1)
        wave_vector_here = self._wave_vector(index)
        here = self._grid[index]
        prefactor_diagonal = (wave_vector_prev + wave_vector_here)/(2*wave_vector_prev)
        prefactor_off_diagonal = (wave_vector_prev - wave_vector_here)/(2*wave_vector_prev)
        upper_left = prefactor_diagonal*np.exp(-(wave_vector_prev-wave_vector_here)*here)
        upper_right = prefactor_off_diagonal*np.exp(-(wave_vector_prev+wave_vector_here)*here)
        lower_left = prefactor_off_diagonal*np.exp((wave_vector_here+wave_vector_prev)*here)
        lower_right = prefactor_diagonal*np.exp((wave_vector_prev-wave_vector_here)*here)
        return np.array([[upper_left, upper_right],
                         [lower_left, lower_right]])


class ScatterMatrixSolver(BaseMatrixSolver):

    def __init__(self, grid, potential, energy,):
        super().__init__(grid, potential, energy)
        self._matrix = ScatterMatrix

    # private
    def _product(self, first, second):
        return np.array([[first[0, 0] + first[0, 1]*second[0, 0]*first[1, 0]/(1 - first[1, 1]*second[0, 0]),
                          first[0, 1]*second[0, 1]/(1 - second[0, 0]*first[1, 1])],
                         [second[1, 0]*first[1, 0]/(1 - first[1, 1]*second[0, 0]),
                          second[1, 1]*second[1, 0]*first[1, 1]*second[0, 1]/(1 - first[1, 1]*second[0, 0])]])

    def _matrix_factor(self, index):
        return 0


class ScatterMatrix(BaseMatrix):

    def __init__(self, value=None):
        super().__init__(value=value)


class TransferMatrix(BaseMatrix):

    def __init__(self, value=None):
        super().__init__(value=value)

    def transmission(self):
        left = 1/np.abs(self._value[0, 0])**2
        right = np.abs(self._value[1, 1] - self._value[1, 0]*self._value[0, 1]/self._value[0, 0])**2
        return left, right

