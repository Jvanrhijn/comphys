""" This module contains functions to solve the 'Shooting' project

Tests for each function are written at the bottom of the file
"""
import numpy as np
import matplotlib.pyplot as plt
import unittest


def outer_turning_point_newton(effective_potential, energy, grid, guess) -> int:
    """Find index of outer turning point in grid, by Newton's method

    :param effective_potential: effective potential energy function
    :param energy: energy eigenvalue of particle
    :param grid: grid for use in numerical computation
    :param guess: initial guess for index, needed for Newton
    :return: index of turning point in grid
    """
    dx = min(np.diff(grid))
    root = find_root_newton(lambda x: effective_potential(x) - energy, 10, grid[guess], dx = dx)
    index = abs(grid - root).argmin()
    return index


def find_root_newton(function, iterations, initial_guess, dx = 0.0001) -> float:
    """Simple implementation of Newton's algorithm for root finding

    :param function: function to find root of
    :param iterations: number of iterations to do
    :param initial_guess: initial guess for root location
    :param dx: step size to use for computing derivative
    :return: location of root, float
    """
    current = initial_guess
    for n in range(0, iterations):
        slope = (function(current + dx) - function(current - dx)) / (2*dx)
        current -= function(current) / slope
    return current


class ShootingTest(unittest.TestCase):
    """ Test cases for the functions in this module

    """
    def find_root_newton(self):
        """Test newton root finding algorithm"""
        self.assertAlmostEqual(find_root_newton(lambda x: (x**2 - 1), 5, 0.5), 1, 3)
        self.assertAlmostEqual(find_root_newton(lambda x: (x**2 - 1), 5, -.5), -1, 3)
        self.assertRaises(TypeError, find_root_newton(lambda x: x, 1, 0.1))

    def test_outer_turning_point(self):
        self.assertEqual(outer_turning_point_newton(lambda x: x**2, 1, np.linspace(0, 10, 1000), 3), 100)
        self.assertEqual(outer_turning_point_newton(lambda x: x**3, 1.5, np.linspace(0, 10, 1000), 3), 114)


if __name__ == '__main__':
    unittest.main()
