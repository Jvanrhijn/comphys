""" This module contains functions to solve the 'Shooting' project

Tests for each function are written at the bottom of the file.
"""
import numpy as np
import matplotlib.pyplot as plt
import unittest


def outer_turning_point_newton(potential, energy, grid, guess) -> int:
    """Find index of outer turning point in grid, by Newton's method

    :param potential: effective potential energy function
    :param energy: energy eigenvalue of particle
    :param grid: grid for use in numerical computation
    :param guess: initial guess for index, needed for Newton
    :return: index of turning point in grid
    """
    dx = min(np.diff(grid))*0.001
    root = find_root_newton(lambda x: potential(x) - energy, 10, grid[guess], dx=dx)
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


def solution_next(previous, pprevious, potential, energy, step_size, *grid_points) -> float:
    """Propagate solution

    :param previous: previous value of solution ('n')
    :param pprevious: previous previous value of solution ('n-1')
    :param potential: effective potential energy function ('W')
    :param energy: energy eigenvalue ('lambda')
    :param step_size: step size ('h')
    :param grid_points: grid points needed for algorithm, in this case only the current point is needed
    :return: value of solution at next grid point
    """
    assert(step_size > 0)
    next_value = 2*previous - pprevious + step_size**2*(potential(grid_points[0]) - energy)*previous
    return next_value


def solution_next_numerov(previous, pprevious, potential, energy, step_size, *grid_points) -> float:
    """Propagate backward solution using Numerov's algorithm

    :param previous: previous value of solution ('n')
    :param pprevious: previous previous value of solution ('n-1')
    :param potential: effective potential energy function ('W')
    :param energy: energy eigenvalue ('lambda')
    :param step_size: step size ('h')
    :param grid_points: grid points needed for algorithm, in this case the current and previous grid point is needed
    :return: value of solution at next (backward) grid point
    """
    assert(step_size > 0)
    next_value = (5*step_size**2/6*(potential(grid_points[0]) - energy) + 2)*previous - (1 - step_size**2/12
                                * (potential(grid_points[1]) - energy))*pprevious
    next_value /= 1 - step_size**2/12*(potential(grid_points[1]) - energy)
    return next_value


def solve_equation_forward(solution_first, solution_second, grid, potential, energy, turning_point_index,
                           numerov=False) -> np.ndarray:
    """Solves the differential equation by forward propagation

    :param solution_first: the solution at the first grid point
    :param solution_second: the solution at the second grid point
    :param grid: the grid to solve the equation on
    :param potential: the potential function to solve the equation for
    :param energy: the energy eigenvalue to solve the equation for
    :param turning_point_index: index of turning point in grid
    :param numerov: use numerov's algorithm for propagation, defaults to False (which uses Euler's algorithm)
    :return: solution (numpy array) obtained by forward propagation
    """
    solution = np.zeros(len(grid))
    solution[0], solution[1] = solution_first, solution_second
    if numerov:
        propagate = solution_next_numerov
    else:
        propagate = solution_next
    for n in range(2, turning_point_index + 2):
        step_size = grid[n] - grid[n-1]
        solution[n] = propagate(solution[n-1], solution[n-2], potential, energy, step_size, grid[n-1], grid[n-2])
    return solution


def solve_equation_backward(solution_last, solution_second_last, grid, potential, energy, turning_point_index,
                            numerov=False) -> np.ndarray:
    """Solves the differential equation by backward propagation

    :param solution_last: value of solution at end point
    :param solution_second_last: value of solution at point prior to end point
    :param grid: grid to solve the equation on
    :param potential: the potential function to solve the equation for
    :param energy: the energy eigenvalue to solve the equation for
    :param turning_point_index: index of turning point in grid
    :param numerov: use Numerov's algorithm for propagation, defaults to False (which uses Euler's algorithm)
    :return: solution (np.ndarray) obtained by forward propagation
    """
    solution = np.zeros(len(grid))
    solution[-1], solution[-2] = solution_last, solution_second_last
    # Choose algorithm
    if numerov:
        propagate = solution_next_numerov
    else:
        propagate = solution_next
    for n in range(len(grid) - 3, turning_point_index - 2, -1):
        step_size = grid[n] - grid[n-1]
        solution[n] = propagate(solution[n+1], solution[n+2], potential, energy, step_size, grid[n+1], grid[n+2])
    return solution


def glue_arrays_together(first_half, second_half, at_index, overwrite=1) -> np.ndarray:
    """Glue two overlapping arrays together at a given index

    :param first_half: first half of array
    :param second_half: second of of array
    :param at_index: index to glue the arrays together at
    :param overwrite: if there is overlap, overwrite entries of this array
    :return: array obtained by gluing the two together
    """
    assert(len(first_half) == len(second_half))
    if overwrite != 1 and overwrite != 2:
        raise(ValueError("Priority kwarg must be either '1' or '2'"))
    length = len(first_half)
    if overwrite == 1:
        for n in range(at_index + 1, length):
            first_half[n] = second_half[n]
        return first_half
    elif overwrite == 2:
        for n in range(length - 1, -1, -1):
            second_half[n] = first_half[n]
        return second_half



def normalize_solution(grid, solution) -> np.ndarray:
    """Normalize wave function

    :param grid: grid to integrate on
    :param solution: non-normalized solution of differential equation
    :return: normalized solution
    """
    integral = np.trapz(solution**2, grid)
    solution /= np.sqrt(integral)
    return solution


class ShootingTest(unittest.TestCase):
    """ Test cases for the functions in this module"""
    def test_find_root_newton(self):
        """Test newton root finding algorithm"""
        self.assertAlmostEqual(find_root_newton(lambda x: (x**2 - 1), 5, 0.5), 1, 3)
        self.assertAlmostEqual(find_root_newton(lambda x: (x**2 - 1), 5, -.5), -1, 3)
        with self.assertRaises(TypeError):
            find_root_newton(lambda x: x, "foo", 0.1)

    def test_outer_turning_point(self):
        self.assertEqual(outer_turning_point_newton(lambda x: x**2, 1, np.linspace(0, 10, 1000), 50), 100)
        self.assertEqual(outer_turning_point_newton(lambda x: x**3, 1.5, np.linspace(0, 10, 1000), 50), 114)

    def test_solution_next(self):
        self.assertAlmostEqual(solution_next(0.1, 0.05, lambda x: 0.5*x**2, 1.5, 0.01, 10), 0.150485, 4)
        self.assertAlmostEqual(solution_next(-0.1, -0.05, lambda x: 0.5*x**2, 1.5, 0.01, 10), -0.150485, 4)
        with self.assertRaises(TypeError):
            solution_next("foo", -0.05, lambda x: 0.5*x**2, 1.5, 0.01, 3)
        with self.assertRaises(AssertionError):
            solution_next(1, 1, lambda x: x, 1, -0.01, 1)
        self.assertEqual(solution_next(0, 0, lambda x: x, 0, 0.01, 0), 0)

    def test_glue_together(self):
        first = np.array([1, 2, 3, 4, 0, 0, 0, 0, 0, 0])
        second = np.array([0, 0, 3, 4, 5, 6, 7, 8, 9, 10])
        np.testing.assert_equal(glue_arrays_together(first, second, 3), np.arange(1, 11))
        np.testing.assert_equal(glue_arrays_together(first, second, 3, overwrite=2), np.arange(1, 11))
        with self.assertRaises(AssertionError):
            glue_arrays_together(np.array([0, 1, 2]), np.array([0, 1]), 1)
        with self.assertRaises(ValueError):
            glue_arrays_together(np.array([0, 1, 2]), np.array([-1, 1, 0]), 1, overwrite=3)

    def test_normalization(self):
        x_axis = np.linspace(0, 1, 100)
        func = x_axis**2
        self.assertAlmostEqual(np.trapz(normalize_solution(x_axis, func)**2, x_axis), 1)


if __name__ == '__main__':
    unittest.main()
