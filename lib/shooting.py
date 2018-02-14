""" This module contains functions to solve the 'Shooting' project

Tests for each function are written at the bottom of the file.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import unittest


def outer_turning_point(potential, energy, grid) -> int:
    """Find index of outer turning point in grid

    :param potential: effective potential energy function
    :param energy: energy eigenvalue of particle
    :param grid: grid to find the turning point on
    :return: index of turning point in grid
    """
    index = abs(potential(grid) - energy).argmin()

    return index


def solution_next(previous, p_previous, potential, energy, step_size, *grid_points) -> float:
    """Propagate solution

    :param previous: previous value of solution ('n')
    :param p_previous: previous previous value of solution ('n-1')
    :param potential: effective potential energy function ('W')
    :param energy: energy eigenvalue ('lambda')
    :param step_size: step size ('h')
    :param grid_points: grid points needed for algorithm, in this case only the current point is needed
    :return: value of solution at next grid point
    """
    assert(step_size > 0)

    return 2 * previous - p_previous + step_size ** 2 * (potential(grid_points[0]) - energy) * previous


def solution_next_log(previous, p_previous, potential, energy, step_size, *grid_points, angular_momentum=0) -> float:
    """Propagate solution using logarithmic grid, now including ugly hack to include angular momentum quantum numberQ

    :param previous: previous value of solution ('n')
    :param p_previous: previous previous value of solution ('n-1')
    :param potential: effective potential energy function ('W')
    :param energy: energy eigenvalue ('lambda')
    :param step_size: step size ('h')
    :param grid_points: grid points needed for algorithm, in this case only the current point is needed
    :return: value of solution at next grid point
    """

    # Transform step size for the propagation algorithm
    step_size /= grid_points[0]

    return (2 * previous - p_previous + step_size ** 2 * ((0.5 + angular_momentum)**2 +
                                                          ((potential(grid_points[0]) - energy)
                                                          * grid_points[0]**2)) * previous)


def solution_next_log_numerov(previous, p_previous, potential, energy, step_size, *grid_points, angular_momentum=0) -> float:
    """Propagate solution using logarithmic grid and Numerov's algorithm, now including ugly hack to include
    angular momentum quantum number!

    :param previous: previous value of solution ('n')
    :param p_previous: previous previous value of solution ('n-1')
    :param potential: effective potential energy function ('W')
    :param energy: energy eigenvalue ('lambda')
    :param step_size: step size ('h')
    :param grid_points: grid points needed for algorithm, in this case only the current point is needed
    :param angular_momentum: angular momentum quantum number
    :return: value of solution at next grid point
    """

    # Transform step size for the propagation algorithm
    step_size /= grid_points[0]

    # Kind of ugly and inefficient to define this here, but at least it keeps the notation tidy
    def q(grid_point):
        return 1 - step_size**2/12*((angular_momentum + 0.5)**2 + (potential(grid_point) - energy)*grid_point**2)

    next_value = (12 - 10*q(grid_points[0]))*previous - q(grid_points[1])*p_previous
    next_value /= q(grid_points[2])
    return next_value


def solution_next_numerov(previous, p_previous, potential, energy, step_size, *grid_points) -> float:
    """Propagate solution using Numerov's algorithm

    :param previous: previous value of solution ('n')
    :param p_previous: previous previous value of solution ('n-1')
    :param potential: effective potential energy function ('W')
    :param energy: energy eigenvalue ('lambda')
    :param step_size: step size ('h')
    :param grid_points: grid points needed for algorithm, in this case the current and previous grid point are needed
    :return: value of solution at next (backward) grid point
    """
    assert(step_size > 0)

    next_value = (5*step_size**2/6*(potential(grid_points[0]) - energy) + 2) * previous - (
            1 - step_size**2/12
            * (potential(grid_points[1]) - energy)) * p_previous
    next_value /= 1 - step_size**2/12*(potential(grid_points[2]) - energy)

    return next_value


def solve_equation_forward(solution_first, solution_second, grid, potential, energy, turning_point_index,
                           propagate) -> np.ndarray:
    """Solves the differential equation by forward propagation

    :param solution_first: the solution at the first grid point
    :param solution_second: the solution at the second grid point
    :param grid: the grid to solve the equation on
    :param potential: the potential function to solve the equation for
    :param energy: the energy eigenvalue to solve the equation for
    :param turning_point_index: index of turning point in grid
    :param propagate: propagation function to use, should return next value of solution given previous values and grid
           points
    :return: solution (numpy array) obtained by forward propagation
    """
    # Check that the turning point is internal to the domain
    assert(turning_point_index not in [0, len(grid) - 1])

    solution = np.zeros(len(grid))
    solution[0], solution[1] = solution_first, solution_second

    for n in range(2, turning_point_index + 2):
        step_size = grid[n] - grid[n-1]
        solution[n] = propagate(solution[n-1], solution[n-2], potential, energy, step_size,
                                grid[n-1], grid[n-2], grid[n])

    return solution


def solve_equation_backward(solution_last, solution_second_last, grid, potential, energy, turning_point_index,
                            propagate) -> np.ndarray:
    """Solves the differential equation by backward propagation

    :param solution_last: value of solution at end point
    :param solution_second_last: value of solution at point prior to end point
    :param grid: grid to solve the equation on
    :param potential: the potential function to solve the equation for
    :param energy: the energy eigenvalue to solve the equation for
    :param turning_point_index: index of turning point in grid
    :param propagate: propagation function to use, should return next value of solution given previous values and grid
           points
    :return: solution (numpy array) obtained by forward propagation
    """
    # Check that the turning point is internal to the domain
    assert(turning_point_index not in [0, len(grid) - 1])

    solution = np.zeros(len(grid))
    solution[-1], solution[-2] = solution_last, solution_second_last

    for n in range(len(grid) - 3, turning_point_index - 2, -1):
        step_size = grid[n] - grid[n-1]
        solution[n] = propagate(solution[n+1], solution[n+2], potential, energy, step_size,
                                grid[n+1], grid[n+2], grid[n])
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
        raise(ValueError("Overwrite kwarg must be either '1' or '2'"))

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
    """Normalize wave function using trapezoidal integration

    :param grid: grid to integrate on
    :param solution: non-normalized solution of differential equation
    :return: normalized solution
    """
    integral = np.trapz(solution**2, grid)
    solution /= np.sqrt(integral)

    return solution


def solve_equation(solution_first, solution_second, solution_last, solution_second_last, grid, potential, energy,
                   turning_point_index, propagate) -> np.ndarray:
    """Solve the differential equation on the entire grid

    :param solution_first: first value of solution
    :param solution_second: second value of solution
    :param solution_last: last value of solution
    :param solution_second_last: second last value of solution
    :param grid: grid to solve equation on
    :param potential: potential to solve equation for
    :param energy: energy eigenvalue guess
    :param turning_point_index: index of turning point in grid
    :param propagate: propagation function to use, should return next value of solution given previous values and grid
           points
    :return: normalized numerical solution of radial Schrödinger equation on grid,
    """
    solution_forward = solve_equation_forward(solution_first, solution_second, grid, potential, energy,
                                              turning_point_index, propagate)
    solution_backward = solve_equation_backward(solution_last, solution_second_last, grid, potential, energy,
                                                turning_point_index, propagate)

    # Match forward and backward solutions and normalize
    solution_forward /= solution_forward[turning_point_index]
    solution_backward /= solution_backward[turning_point_index]
    solution = glue_arrays_together(solution_forward, solution_backward, turning_point_index)
    solution = normalize_solution(grid, solution)

    return solution


def continuity_measure_function(solution_left, solution_right, turning_point) -> float:
    """Function F(lambda) from the lecture notes

    :param solution_left: solution obtained by forward propagation
    :param solution_right: solution obtained by backward propagation
    :param turning_point: turning point index
    :return: derivative continuity measure at turning point
    """
    return (solution_right[turning_point+1] - solution_right[turning_point-1]
            - (solution_left[turning_point+1] - solution_left[turning_point-1]))


def shooting_iteration_bisection(grid, solution_first, solution_second, solution_last, solution_second_last,
                                 left_bound, right_bound, potential, propagate):
    """Do an iteration of the bisection method for finding a better approximation of the eigenvalue

    :param grid: grid to solve equation on
    :param solution_first: first value of solution
    :param solution_second: second value of solution
    :param solution_last: last value of solution
    :param solution_second_last: second last value of solution
    :param left_bound: left bound of bisection interval
    :param right_bound: right bound of bisection interval
    :param potential: potential energy function to solve equation for
    :param propagate: propagation function to use, should return next value of solution given previous values and grid
           points
    :return: new interval to bisect
    """
    mid_point = 0.5*(right_bound + left_bound)
    turning_point_left = outer_turning_point(potential, left_bound, grid)
    turning_point_mid = outer_turning_point(potential, mid_point, grid)

    # Calculate forward & backward solutions for the lower bound & the interval midpoint
    solution_forward_left = solve_equation_forward(solution_first, solution_second, grid, potential, left_bound,
                                                   turning_point_left, propagate)
    solution_backward_left = solve_equation_backward(solution_last, solution_second_last, grid, potential, left_bound,
                                                     turning_point_left, propagate)
    solution_forward_mid = solve_equation_forward(solution_first, solution_second, grid, potential, mid_point,
                                                  turning_point_mid, propagate)
    solution_backward_mid = solve_equation_backward(solution_first, solution_second, grid, potential, mid_point,
                                                    turning_point_mid, propagate)

    # Match the forward and backward solutions at the turning point
    solution_forward_left /= solution_forward_left[turning_point_left]
    solution_backward_left /= solution_backward_left[turning_point_left]
    solution_backward_mid /= solution_backward_mid[turning_point_mid]
    solution_forward_mid /= solution_forward_mid[turning_point_mid]

    left_derivative_continuity = continuity_measure_function(solution_forward_left, solution_backward_left,
                                                             turning_point_left)
    mid_derivative_continuity = continuity_measure_function(solution_forward_mid, solution_backward_mid,
                                                            turning_point_mid)

    # Narrow down the interval containing the root according to the bisection algorithm
    if np.sign(left_derivative_continuity) == np.sign(mid_derivative_continuity):
        return mid_point, right_bound, mid_derivative_continuity
    else:
        return left_bound, mid_point, mid_derivative_continuity


def shooting_iteration_improved(solution_first, solution_second, solution_last, solution_second_last, grid, potential,
                                eigenvalue_guess, tolerance, propagate):
    """Retrieve next eigenvalue guess using the improved algorithm

    :param solution_first: first value of solution
    :param solution_second: second value of solution
    :param solution_last: last value of solution
    :param solution_second_last: second last value of solution
    :param grid: grid to sove equation on
    :param potential: potential function to solve equation for
    :param eigenvalue_guess: guess for eigenvalue
    :param tolerance: tolerance to check for convergence
    :param propagate: propagation function to use, should return next value of solution given previous values and grid
           points
    :return: new eigenvalue guess, measure of derivative continuity, whether the algorithm has converged sufficiently
    """
    turning_point = outer_turning_point(potential, eigenvalue_guess, grid)

    solution_left = solve_equation_forward(solution_first, solution_second,
                                           grid, potential, eigenvalue_guess,
                                           turning_point, propagate)
    solution_right = solve_equation_backward(solution_last, solution_second_last,
                                             grid, potential, eigenvalue_guess,
                                             turning_point, propagate)

    solution_left /= solution_left[turning_point]
    solution_right /= solution_right[turning_point]

    # Use trapezoidal integration to calculate factor 'A'
    # Now including ugly hack to make a special case for a logarithmic grid!
    if propagate == solution_next_log:
        factor = (np.trapz(grid[:turning_point]**2*solution_left[:turning_point]**2,
                           np.log(grid[:turning_point]))
                  + np.trapz(grid[turning_point:]**2*solution_right[turning_point:]**2,
                             np.log(grid[turning_point:])))**-1
    else:
        factor = (np.trapz(solution_left[:turning_point]**2, grid[:turning_point])
                  + np.trapz(solution_right[turning_point:]**2, grid[turning_point:]))**-1
    derivative_continuity = continuity_measure_function(solution_left, solution_right, turning_point)

    # Get next eigenvalue guess
    step_size = np.diff(grid)[turning_point]
    eigenvalue_guess -= factor*derivative_continuity/(2*step_size)

    # Check for convergence
    converged = (abs(derivative_continuity*factor) < tolerance)

    return eigenvalue_guess, derivative_continuity, converged


def shooting_method(grid, solution_first, solution_second, solution_last, solution_second_last,
                    tolerance, max_iterations, potential, propagate, *algorithm_inputs,
                    algorithm='bisection'):
    """Calculate an eigenvalue of the Schrödinger equation for a given potential, using the shooting method

    :param grid: grid to apply method on
    :param solution_first: solution at first grid point
    :param solution_second: solution at second grid point
    :param solution_last: solution at last grid point
    :param solution_second_last: solution at second last grid point
    :param tolerance: convergence tolerance
    :param max_iterations: maximum iterations to run
    :param potential: potential function to solve equation for
    :param algorithm_inputs: inputs required for algorithm, bracket for bisection, initial guess for improved algorithm
    :param algorithm: algorithm to use for finding roots, either 'bisection' or 'improved'
    :param propagate: propagation function to use, should return next value of solution given previous values and grid
           points
    :return: eigenvalue obtained using the shooting method
    """
    assert(algorithm == 'bisection' or algorithm == 'improved')
    iterations = 0
    continuity_iterations, eigenvalues = [], []
    converged = False

    if algorithm == 'bisection':
        left_bound = algorithm_inputs[0]
        right_bound = algorithm_inputs[1]

        while not converged and iterations < max_iterations:
            old_root_guess = 0.5*(right_bound + left_bound)
            left_bound, right_bound, mid_continuity = shooting_iteration_bisection(
                                                                                grid, solution_first,
                                                                                solution_second, solution_last,
                                                                                solution_second_last, left_bound,
                                                                                right_bound, potential,
                                                                                propagate
                                                                                )
            new_root_guess = 0.5*(right_bound + left_bound)

            continuity_iterations.append(mid_continuity)
            eigenvalues.append(new_root_guess)

            iterations += 1
            converged = (abs(new_root_guess - old_root_guess) < tolerance)
    elif algorithm == 'improved':
        eigenvalue_guess = algorithm_inputs[0]

        while not converged and iterations < max_iterations:
            eigenvalue_guess, derivative_continuity, converged = shooting_iteration_improved(
                                                                                  solution_first, solution_second,
                                                                                  solution_last, solution_second_last,
                                                                                  grid, potential, eigenvalue_guess,
                                                                                  tolerance, propagate
                                                                                  )
            continuity_iterations.append(derivative_continuity)
            eigenvalues.append(eigenvalue_guess)

            iterations += 1

    return eigenvalues, continuity_iterations


def generate_latex_table(left_column, right_column, header_left, header_right, rounding=None) -> str:
    """Generate a LaTeX table from two lists

    :param left_column: list with values to list in left column
    :param right_column: list with values to list in right column
    :param header_left: left header
    :param header_right: right header
    :param rounding: digits to round values to, defaults to no rounding
    :return: string containing LaTeX formatted table
    """
    table = "{0} & {1}\\\\\\hline\n".format(header_left, header_right)

    for left, right in zip(left_column, right_column):
        if rounding is not None:
            left = round(left, rounding)
            right = round(right, rounding)
        table += str(left) + " & " + str(right) + "\\\\ \n"

    return table


class ShootingTest(unittest.TestCase):
    """ Test cases for the functions in this module"""
    def test_outer_turning_point(self):
        self.assertEqual(outer_turning_point(lambda x: x**2, 1, np.linspace(0, 2, 200)), 99)
        self.assertEqual(outer_turning_point(lambda x: x**3, 1.5, np.linspace(0, 2, 200)), 114)

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
        self.assertAlmostEqual(np.trapz(normalize_solution(x_axis, func)**2, x_axis), 1, 5)


if __name__ == '__main__':
    unittest.main()