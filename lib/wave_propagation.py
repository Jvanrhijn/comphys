"""This module contains classes needed for the Wave Propagation project"""
import numpy as np


class BaseMatrix:
    """Base quantum matrix class"""
    def __init__(self, *args):
        if len(args) != 0:
            if args[0].shape != (2, 2):
                raise ValueError("Matrix dimension must be 2x2")
            self._value = args[0]
        else:
            self._value = np.identity(2)

    @property
    def value(self):
        """Return value of matrix as np.ndarray"""
        return self._value

    def __getitem__(self, key):
        return self._value[key]

    def __setitem__(self, key, value):
        self._value[key] = value

    def __matmul__(self, other):
        return 0

    def __rmatmul__(self, other):
        return 0

    def transmission(self):
        """Get transmission coefficients of unit velocity wave incoming from left and right, respectively"""
        return 0, 0

    def reflection(self):
        """Get reflection coefficients of unit wave incoming from left and right, respectively"""
        pass


class BaseMatrixSolver:
    """Base matrix solver class, overridden by scattering/transfer matrix solvers"""
    def __init__(self, grid, potential, energy):
        self._grid = grid
        self._energy = energy
        self._num_factors = len(self._grid)
        grid_diff = np.concatenate((np.diff(grid), np.zeros(1)))
        self._potential = potential(grid+grid_diff/2)
        self._Matrix = BaseMatrix
        self._result: BaseMatrix = None

    @property
    def matrix(self):
        assert self._result is not None
        return self._result

    def calculate(self):
        """Calculate the matrix representing the potential barrier/well.
        return
        """
        total_product = self._Matrix()
        for index in range(1, self._num_factors):
            total_product = total_product @ self._matrix_factor(index)
        self._result = total_product
        return total_product

    def transmission(self):
        """Calculate transmission coefficients of wave coming in from left and right of potential barriers"""
        assert self._result is not None
        velocity_left = np.sqrt(abs(self._energy - self._potential[0]))
        velocity_right = np.sqrt(abs(self._energy - self._potential[-1]))
        transmission_left, transmission_right = self._result.transmission()
        return transmission_left*velocity_right/velocity_left, transmission_right*velocity_left/velocity_right

    def _wave_vector(self, index):
        """Calculate the local wave vector (eta_i in the lecture notes)"""
        potential = self._potential[index]
        if self._energy <= potential:
            wave_vector = np.sqrt(potential - self._energy)
        else:
            wave_vector = np.sqrt(self._energy - potential)*1j
        return wave_vector

    def _matrix_factor(self, index):
        """Calculate matrix factor [index] in total product"""
        pass


class ScatterMatrix(BaseMatrix):
    """Scattering matrix class, implements multiplication rule & physical quantity calculation"""
    def __init__(self, *args):
        super().__init__(*args)
        if len(args) == 0:
            self._value = np.array([[0, 1], [1, 0]])

    def __matmul__(self, other: BaseMatrix):
        """Multiplication rule for scatter matrices"""
        first = self.value
        second = other.value
        return ScatterMatrix(
            np.array([[first[0, 0]
                       + first[0, 1]*second[0, 0]*first[1, 0]/(1 - first[1, 1]*second[0, 0]),
                       first[0, 1]*second[0, 1]/(1 - second[0, 0]*first[1, 1])],
                      [second[1, 0]*first[1, 0]/(1 - first[1, 1]*second[0, 0]),
                       second[1, 1]
                       + second[1, 0]*first[1, 1]*second[0, 1]/(1 - first[1, 1]*second[0, 0])]]))

    def transmission(self):
        left = np.abs(self._value[1, 0])**2
        right = np.abs(self._value[0, 1])**2
        return left, right


class TransferMatrix(BaseMatrix):
    """Transfer matrix class, implements multiplication rule & physical quantity calculation"""
    def __init__(self, *args):
        super().__init__(*args)

    def __matmul__(self, other: BaseMatrix):
        return TransferMatrix(self.value @ other.value)

    def transmission(self):
        left = 1/np.abs(self._value[0, 0])**2
        right = np.abs(self._value[1, 1] - self._value[1, 0]*self._value[0, 1]/self._value[0, 0])**2
        return left, right


class TransferMatrixSolver(BaseMatrixSolver):
    """Transfer matrix class, overrides BaseMatrixSolver"""
    def __init__(self, grid, potential, energy):
        super().__init__(grid, potential, energy)
        self._Matrix = TransferMatrix

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
        return TransferMatrix(np.array([[upper_left, upper_right],
                                              [lower_left, lower_right]]))


class ScatterMatrixSolver(BaseMatrixSolver):
    """Scattering matrix solver class, overrides BaseMatrixSolver"""
    def __init__(self, grid, potential, energy,):
        super().__init__(grid, potential, energy)
        self._Matrix = ScatterMatrix

    # private
    def _matrix_factor(self, index):
        here = self._grid[index]
        wave_vector_prev = self._wave_vector(index-1)
        wave_vector_here = self._wave_vector(index)
        prefactor_diagonal = (wave_vector_prev - wave_vector_here)/(wave_vector_prev + wave_vector_here)
        upper_left = prefactor_diagonal*np.exp(2*wave_vector_prev*here)
        upper_right = 2*wave_vector_here/(wave_vector_here+wave_vector_prev)\
            * np.exp((wave_vector_prev - wave_vector_here)*here)
        lower_left = 2*wave_vector_prev/(wave_vector_prev + wave_vector_here)\
            * np.exp((wave_vector_prev - wave_vector_here)*here)
        lower_right = -prefactor_diagonal*np.exp(-2*wave_vector_here*here)
        return ScatterMatrix(np.array([[upper_left, upper_right],
                                       [lower_left, lower_right]]))

