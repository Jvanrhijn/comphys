"""Tests for wave propagation module"""
import unittest
import numpy as np
import lib.wave_propagation as wp


def potential_square(x):
    return np.heaviside(x, 1) - np.heaviside(x - 1, 1)


class TestTransferMat(unittest.TestCase):

    def test_product(self):
        grid = np.arange(0, 1,  10)
        transfer_matrix = wp.TransferMatrixSolver(grid, potential_square, 0.5)
        first = np.random.randint(0, 10, size=(2, 2))
        second = np.random.randint(0, 10, size=(2, 2))
        np.testing.assert_array_equal(transfer_matrix._product(first, second), np.dot(first, second))

    def test_solve(self):
        self.assertTrue(False)

    def test_submatrices(self):
        self.assertTrue(False)


class TestScatterMat(unittest.TestCase):

    def test_product(self):
        grid = np.arange(0, 1, 10)
        first = np.random.randint(0, 10, size=(2, 2))
        second = np.random.randint(0, 10, size=(2, 2))
        scatter_matrix = wp.ScatterMatrixSolver(grid, potential_square, 0.5)
        result_00 = first[0, 0] + first[0, 1]*second[0, 0]*first[1, 0]/(1 - first[1, 1]*second[0, 0])
        result_01 = first[0, 1]*second[0, 1]/(1 - second[0, 0]*first[1, 1])
        result_10 = second[1, 0]*first[1, 0]/(1 - first[1, 1]*second[0, 0])
        result_11 = second[1, 1]*second[1, 0]*first[1, 1]*second[0, 1]/(1 - first[1, 1]*second[0, 0])
        result = np.array([[result_00, result_01], [result_10, result_11]])
        np.testing.assert_array_equal(scatter_matrix._product(first, second), result)

    def test_factor(self):
        self.assertTrue(False)

    def test_solve(self):
        self.assertTrue(False)


if __name__ == "__main__":
    suite_transfer = unittest.TestLoader().loadTestsFromTestCase(TestTransferMat)
    suite_scatter = unittest.TestLoader().loadTestsFromTestCase(TestScatterMat)
    unittest.TextTestRunner(verbosity=2).run(suite_transfer)
    unittest.TextTestRunner(verbosity=2).run(suite_scatter)
