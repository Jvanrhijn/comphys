import sys
import lib.shooting as shooting
import numpy as np
from decorators.decorators import *
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


@single_plot
def excitons1b():
    energies = np.linspace(0, 10, 1000)
    grid = np.linspace(0, 10, 1000)
    turning_points = np.zeros(len(grid))
    for n, energy in enumerate(energies):
        turning_points[n] = grid[shooting.outer_turning_point(lambda x: 0.25*x**2, energy, grid)]
    return energies, turning_points, r"$\lambda$", "Turning point"


@plot_grid_show
def excitons1c():
    def potential(x):
        return 0.25*x**2
    # Eigenvalue guess
    eigenvalue_guess = 1.5
    # Set up equidistant grid
    grid_points = 100
    grid_displacement = 0
    grid_end = 5
    grid = np.linspace(0, grid_end, grid_points) + grid_displacement
    # Calculate analytic solution
    analytic_solution = grid*np.exp(-grid**2/4)
    analytic_solution = shooting.normalize_solution(grid, analytic_solution)
    # Get classical turning point index
    turning_point = shooting.outer_turning_point(potential, eigenvalue_guess, grid)
    # Set up initial values of forward and backward solutions
    solution_forward_first = grid[0]
    solution_forward_second = grid[1]
    solution_backward_last = analytic_solution[-1]
    solution_backward_second_last = analytic_solution[-2]
    # Solve the differential equation
    solution_forward = shooting.solve_equation_forward(solution_forward_first, solution_forward_second,
                                                       grid, potential, eigenvalue_guess, turning_point)
    solution_backward = shooting.solve_equation_backward(solution_backward_last, solution_backward_second_last,
                                                         grid, potential, eigenvalue_guess, turning_point)
    # Match the solutions at the turning point
    solution_forward /= solution_forward[turning_point]
    solution_backward /= solution_backward[turning_point]
    # Glue solutions together
    solution = shooting.glue_arrays_together(solution_forward, solution_backward, turning_point)
    # Normalize solution
    solution = shooting.normalize_solution(grid, solution)
    # Plot both solutions and analytic solution
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(grid, analytic_solution, label="Analytic solution")
    ax[0].plot(grid[0:turning_point+1], solution[0:turning_point+1], label="Forward solution")
    ax[0].plot(grid[turning_point:], solution[turning_point:], label="Backward solution")
    ax[1].set_xlabel(r"$\rho$")
    ax[0].set_ylabel(r"$\zeta(\rho)$")
    ax[1].set_ylabel(r"$\zeta(\rho) - \zeta_{00}(\rho)$, $10^{-4}$")
    # Plot relative error
    error = solution - analytic_solution
    ax[1].plot(grid, error*10**4)
    ax[0].legend()
    return ax


@vertical_subplots
def excitons2a():
    def potential(x):
        return 0.25*x**2
    # Eigenvalue guess
    eigenvalue_guess = 1.5
    # Set up equidistant grid
    grid_points = 100
    grid_displacement = 0
    grid_end = 5
    grid = np.linspace(0, grid_end, grid_points) + grid_displacement
    # Calculate analytic solution
    analytic_solution = grid*np.exp(-grid**2/4)
    analytic_solution = shooting.normalize_solution(grid, analytic_solution)
    # Get classical turning point index
    turning_point = shooting.outer_turning_point(potential, eigenvalue_guess, grid)
    # Set up initial values of forward and backward solutions
    solution_first = grid[0]
    solution_second = grid[1]
    solution_last = analytic_solution[-1]
    solution_second_last = analytic_solution[-2]
    # Solve the differential equation
    solution_numerov = shooting.solve_equation(solution_first, solution_second, solution_last,
                                               solution_second_last, grid, potential, eigenvalue_guess,
                                               turning_point, numerov=True)
    error_numerov = solution_numerov - analytic_solution
    return grid, solution_numerov, error_numerov, r"$\rho", r"$\zeta(\rho)", r"$\zeta(\rho) - \zeta_{00}(\rho)$, $10^4$"


@single_plot
def excitons2b():
    def potential(x):
        return 0.25*x**2
    left_bound = 0.3
    right_bound = 1.8
    # Set up equidistant grid
    grid_points = 100
    grid_displacement = 0
    grid_end = 5
    grid = np.linspace(0, grid_end, grid_points) + grid_displacement
    # Calculate analytic solution
    analytic_solution = grid*np.exp(-grid**2/4)
    analytic_solution = shooting.normalize_solution(grid, analytic_solution)
    # Set up initial values of forward and backward solutions
    solution_first = grid[0]
    solution_second = grid[1]
    solution_last = analytic_solution[-1]
    solution_second_last = analytic_solution[-2]
    # Do shooting method
    tolerance = 10**-4
    max_iterations = 100
    eigenvalue = shooting.shooting_method(grid, solution_first, solution_second, solution_last, solution_second_last,
                                          tolerance, max_iterations, potential, left_bound, right_bound, numerov=True)
    # Get classical turning point index
    turning_point = shooting.outer_turning_point(potential, eigenvalue, grid)
    # Solve equation for found eigenvalue
    solution = shooting.solve_equation(solution_first, solution_second, solution_last, solution_second_last, grid,
                                       potential, eigenvalue, turning_point, numerov=True)
    return grid, solution, r"$\rho$", r"$\zeta(\rho)$"
