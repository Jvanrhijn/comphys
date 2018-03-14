"""Tests for wave propagation module"""
import unittest
import numpy as np
import lib.wave_propagation as wp


def potential_square(x):
    return np.heaviside(x + 0.5, 1) - np.heaviside(x - 0.5, 1)


class TestTransferMat(unittest.TestCase):

    def test_product(self):
        grid = np.linspace(0, 1, 11)
        transfer_matrix = wp.TransferMatrixSolver(grid, potential_square, 0.5)
        first = np.random.randint(0, 10, size=(2, 2))
        second = np.random.randint(0, 10, size=(2, 2))
        np.testing.assert_array_equal(transfer_matrix._product(first, second), np.dot(first, second))

    def test_factors(self):
        energy = 0
        grid = np.linspace(0, 1, 11)
        transfer_matrix = wp.TransferMatrixSolver(grid, potential_square, energy)
        p_matrix_expected = np.array([[1, 0],
                                      [0, 1]])
        q_matrix_expected = np.array([[np.e**0.1, 0],
                                      [0, 1/np.e**0.1]])
        np.testing.assert_array_almost_equal(p_matrix_expected, transfer_matrix._p_submatrix(1))
        np.testing.assert_array_almost_equal(q_matrix_expected, transfer_matrix._q_submatrix(1))

    def test_solve(self):
        grid = np.linspace(-0.5, 0.5, 100)
        width_barrier = 1
        energy = 0.5
        k, eta = np.sqrt(energy), np.sqrt(1 - energy)
        T_analytical = (1 + ((k**2 + eta**0.5)/(2*k*eta))*np.sinh(eta*width_barrier))**-1
        transfer_solver = wp.TransferMatrixSolver(grid, potential_square, energy)
        transmission = transfer_solver.calculate()
        self.assertAlmostEqual(transmission, T_analytical)


class TestScatterMat(unittest.TestCase):

    def test_product(self):
        grid = np.linspace(0, 1, 11)
        first = np.random.randint(0, 10, size=(2, 2))
        second = np.random.randint(0, 10, size=(2, 2))
        scatter_matrix = wp.ScatterMatrixSolver(grid, potential_square, 0.5)
        result_00 = first[0, 0] + first[0, 1]*second[0, 0]*first[1, 0]/(1 - first[1, 1]*second[0, 0])
        result_01 = first[0, 1]*second[0, 1]/(1 - second[0, 0]*first[1, 1])
        result_10 = second[1, 0]*first[1, 0]/(1 - first[1, 1]*second[0, 0])
        result_11 = second[1, 1]*second[1, 0]*first[1, 1]*second[0, 1]/(1 - first[1, 1]*second[0, 0])
        result = np.array([[result_00, result_01], [result_10, result_11]])
        np.testing.assert_array_equal(scatter_matrix._product(first, second), result)

    def test_factors(self):
        energy = 0
        grid = np.linspace(0, 1, 11)
        scatter_matrix = wp.ScatterMatrixSolver(grid, potential_square, energy)
        factor_expected = np.array([[0, 1],
                                    [1, 0]])
        np.testing.assert_array_almost_equal(factor_expected, scatter_matrix._matrix_factor(1))

    def test_solve(self):
        grid = np.linspace(-0.5, 0.5, 100)
        width_barrier = 1
        energy = 0.5
        k, eta = np.sqrt(energy), np.sqrt(1 - energy)
        T_analytical = (1 + ((k**2 + eta**0.5)/(2*k*eta))*np.sinh(eta*width_barrier))**-1
        transfer_solver = wp.ScatterMatrixSolver(grid, potential_square, energy)
        transmission = transfer_solver.calculate()
        self.assertAlmostEqual(transmission, T_analytical)


if __name__ == "__main__":
    x = np.linspace(-1, 1, 1000)
    plt.figure()
    plt.plot(x, potential_square(x))
    plt.show()
    suite_transfer = unittest.TestLoader().loadTestsFromTestCase(TestTransferMat)
    suite_scatter = unittest.TestLoader().loadTestsFromTestCase(TestScatterMat)
    unittest.TextTestRunner(verbosity=2).run(suite_transfer)
    unittest.TextTestRunner(verbosity=2).run(suite_scatter)
