"""Tests for wave propagation module"""
import unittest
import numpy as np
import lib.wave_propagation as wp


def potential_square(x):
    return np.heaviside(x + 0.5, 1) - np.heaviside(x - 0.5, 1)


class TestTransferMatSolver(unittest.TestCase):

    def test_product(self):
        first = wp.TransferMatrix(value=np.random.randint(0, 10, size=(2, 2)))
        second = wp.TransferMatrix(value=np.random.randint(0, 10, size=(2, 2)))
        np.testing.assert_array_equal((first @ second)._value, np.dot(first._value, second._value))

    def test_factors(self):
        energy = 0
        grid = np.linspace(0, 0.5, 11)
        transfer_matrix = wp.TransferMatrixSolver(grid, potential_square, energy)
        factor_expected = np.array([[1, 0],
                                    [0, 1]])
        np.testing.assert_array_almost_equal(factor_expected, transfer_matrix._matrix_factor(1)._value)

    def test_solve(self):
        grid = np.array([-1] + list(np.linspace(-0.5, 0.5, 2)))
        width_barrier = 1
        energy = 0.5
        k, eta = np.sqrt(energy), np.sqrt(1 - energy)
        t_analytical = (1 + ((k**2 + eta**2)/(2*k*eta))**2*np.sinh(eta*width_barrier)**2)**-1
        transmission = wp.TransferMatrixSolver(grid, potential_square, energy).calculate().transmission()[0]
        self.assertAlmostEqual(transmission, t_analytical)


class TestScatterMatSolver(unittest.TestCase):

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
        grid = np.linspace(-0.5001, 0.5, 100)
        width_barrier = 1
        energy = 0.5
        k, eta = np.sqrt(energy), np.sqrt(1 - energy)
        t_analytical = (1 + ((k**2 + eta**2)/(2*k*eta))**2*np.sinh(eta*width_barrier))**-1
        scatter_solver = wp.ScatterMatrixSolver(grid, potential_square, energy)
        transmission = scatter_solver.calculate().transmission()[0]
        self.assertAlmostEqual(transmission, t_analytical)


class TestTransferMat(unittest.TestCase):

    def test_init(self):
        values = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            wp.TransferMatrix(value=values)

    def test_transmission(self):
        transfer_matrix = wp.TransferMatrix(value=np.ones((2, 2)))
        self.assertEqual(transfer_matrix.transmission()[0], 1)
        self.assertEqual(transfer_matrix.transmission()[1], 0)


if __name__ == "__main__":
    suite_transfer = unittest.TestLoader().loadTestsFromTestCase(TestTransferMatSolver)
    suite_scatter = unittest.TestLoader().loadTestsFromTestCase(TestScatterMatSolver)
    suite_trmat = unittest.TestLoader().loadTestsFromTestCase(TestTransferMat)
    unittest.TextTestRunner(verbosity=2).run(suite_transfer)
    unittest.TextTestRunner(verbosity=2).run(suite_scatter)
    unittest.TextTestRunner(verbosity=2).run(suite_trmat)
