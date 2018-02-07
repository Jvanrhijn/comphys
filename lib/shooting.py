""" This module contains functions to solve the 'Shooting' project

Tests for each function are written at the bottom of the file.
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
    dx = min(np.diff(grid))*0.001
    root = find_root_newton(lambda x: effective_potential(x) - energy, 10, grid[guess], dx=dx)
    index = abs(grid - root).argmin()
    return index


def find_root_newton(function, iterations, initial_guess, dx=0.0001) -> float:
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


def solution_forward_next(grid_point, previous, pprevious, potential, energy, step_size) -> float:
    """Propagate forward solution

    :param grid_point: previous grid point
    :param previous: previous value of solution ('n')
    :param pprevious: previous previous value of solution ('n-1')
    :param potential: effective potential energy function ('W')
    :param energy: energy eigenvalue ('lambda')
    :param step_size: step size ('h')
    :return: value of solution at next grid point
    """
    assert(step_size > 0)
    next_value = 2*previous - pprevious + step_size**2*(potential(grid_point) - energy)*previous
    return next_value


def solution_backward_next(grid_point, previous, pprevious, potential, energy, step_size) -> float:
    """Propagate backward solution

    :param grid_point: previous grid point
    :param previous: previous value of solution ('n')
    :param pprevious: previous previous value of solution ('n-1')
    :param potential: effective potential energy function ('W')
    :param energy: energy eigenvalue ('lambda')
    :param step_size: step size ('h')
    :return: value of solution at next (backward) grid point
    """
    assert(step_size > 0)
    next_value = 2*previous - pprevious + step_size**2*(potential(grid_point) - energy)*previous
    return next_value


class ShootingTest(unittest.TestCase):
    """ Test cases for the functions in this module

    """
    def find_root_newton(self):
        """Test newton root finding algorithm"""
        self.assertAlmostEqual(find_root_newton(lambda x: (x**2 - 1), 5, 0.5), 1, 3)
        self.assertAlmostEqual(find_root_newton(lambda x: (x**2 - 1), 5, -.5), -1, 3)
        with self.assertRaises(TypeError):
            find_root_newton(lambda x: x, 1, 0.1)

    def test_outer_turning_point(self):
        self.assertEqual(outer_turning_point_newton(lambda x: x**2, 1, np.linspace(0, 10, 1000), 50), 100)
        self.assertEqual(outer_turning_point_newton(lambda x: x**3, 1.5, np.linspace(0, 10, 1000), 50), 114)

    def test_solution_forward_next(self):
        self.assertAlmostEqual(solution_forward_next(10, 0.1, 0.05, lambda x: 0.5*x**2, 1.5, 0.01), 0.150485, 4)
        self.assertAlmostEqual(solution_forward_next(10, -0.1, -0.05, lambda x: 0.5*x**2, 1.5, 0.01), -0.150485, 4)
        with self.assertRaises(TypeError):
            solution_forward_next(3, "foo", -0.05, lambda x: 0.5*x**2, 1.5, 0.01)
        with self.assertRaises(AssertionError):
            solution_forward_next(1, 1, 1, lambda x: x, 1, -0.01)
        self.assertEqual(solution_forward_next(0, 0, 0, lambda x: x, 0, 0.01), 0)

    def test_solution_backward_next(self):
        self.assertAlmostEqual(solution_backward_next(10, 0.1, 0.05, lambda x: 0.5*x**2, 1.5, 0.01), 0.150485, 4)
        self.assertAlmostEqual(solution_backward_next(10, -0.1, -0.05, lambda x: 0.5*x**2, 1.5, 0.01), -0.150485, 4)
        with self.assertRaises(TypeError):
            solution_backward_next(3, "foo", -0.05, lambda x: 0.5*x**2, 1.5, 0.01)
        with self.assertRaises(AssertionError):
            solution_backward_next(1, 1, 1, lambda x: x, 1, -0.01)
        self.assertEqual(solution_backward_next(0, 0, 0, lambda x: x, 0, 0.01), 0)


if __name__ == '__main__':
    unittest.main()
