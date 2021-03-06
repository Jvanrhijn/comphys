"""Tests for wave propagation module"""
import unittest
import numpy as np
import lib.wave_propagation as wp


def potential_square(x):
    return np.heaviside(x + 0.5, 1) - np.heaviside(x - 0.5, 1)


class TestTransferMatSolver(unittest.TestCase):

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
        transmission_left, transmission_right \
            = wp.TransferMatrixSolver(grid, potential_square, energy).calculate().transmission()
        self.assertAlmostEqual(transmission_left, t_analytical)
        self.assertAlmostEqual(transmission_left, transmission_right)


class TestScatterMatSolver(unittest.TestCase):

    def test_factors(self):
        energy = 0
        grid = np.linspace(0, 1, 11)
        scatter_matrix = wp.ScatterMatrixSolver(grid, potential_square, energy)
        factor_expected = np.array([[0, 1],
                                    [1, 0]])
        np.testing.assert_array_almost_equal(factor_expected, scatter_matrix._matrix_factor(1)._value)

    def test_solve(self):
        grid = np.array([-1] + list(np.linspace(-0.5, 0.5, 2)))
        width_barrier = 1
        energy = 0.5
        k, eta = np.sqrt(energy), np.sqrt(1 - energy)
        t_analytical = (1 + ((k**2 + eta**2)/(2*k*eta))**2*np.sinh(eta*width_barrier)**2)**-1
        scatter_solver = wp.ScatterMatrixSolver(grid, potential_square, energy)
        transmission_left, transmission_right = scatter_solver.calculate().transmission()
        self.assertAlmostEqual(transmission_left, t_analytical)
        self.assertAlmostEqual(transmission_left, transmission_right)


class TestTransferMat(unittest.TestCase):

    def test_product(self):
        first = wp.TransferMatrix(np.random.randint(0, 10, size=(2, 2)))
        second = wp.TransferMatrix(np.random.randint(0, 10, size=(2, 2)))
        np.testing.assert_array_equal((first @ second)._value, np.dot(first._value, second._value))

    def test_init(self):
        values = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            wp.TransferMatrix(values)

    def test_transmission(self):
        transfer_matrix = wp.TransferMatrix(np.ones((2, 2)))
        self.assertEqual(transfer_matrix.transmission()[0], 1)
        self.assertEqual(transfer_matrix.transmission()[1], 0)


class TestScatterMat(unittest.TestCase):

    def test_product(self):
        first_mat = wp.ScatterMatrix(np.ones((2, 2))*0.5)
        second_mat = wp.ScatterMatrix(np.ones((2, 2))*0.5)
        result_00 = 2/3
        result_01 = 1/3
        result_10 = 1/3
        result_11 = 2/3
        result = np.array([[result_00, result_01], [result_10, result_11]])
        np.testing.assert_array_equal((first_mat @ second_mat).value, result)

    def test_init(self):
        values = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            wp.ScatterMatrix(values)


if __name__ == "__main__":
    suite_transfer = unittest.TestLoader().loadTestsFromTestCase(TestTransferMatSolver)
    suite_scatter = unittest.TestLoader().loadTestsFromTestCase(TestScatterMatSolver)
    suite_trmat = unittest.TestLoader().loadTestsFromTestCase(TestTransferMat)
    unittest.TextTestRunner(verbosity=2).run(suite_transfer)
    unittest.TextTestRunner(verbosity=2).run(suite_scatter)
    unittest.TextTestRunner(verbosity=2).run(suite_trmat)
