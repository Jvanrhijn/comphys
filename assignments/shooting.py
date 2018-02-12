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
    return grid, solution_numerov, error_numerov, r"$\rho", r"$\zeta(\rho)", r"$\zeta(\rho) - \zeta_{00}(\rho)$"


@plot_single_window
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
    eigenvalues, continuities, = shooting.shooting_method(grid, solution_first, solution_second, solution_last,
                                                          solution_second_last, tolerance, max_iterations, potential,
                                                          left_bound, right_bound, numerov=True)
    eigenvalue = eigenvalues[-1]
    # Get classical turning point index
    turning_point = shooting.outer_turning_point(potential, eigenvalue, grid)
    # Solve equation for found eigenvalue
    solution = shooting.solve_equation(solution_first, solution_second, solution_last, solution_second_last, grid,
                                       potential, eigenvalue, turning_point, numerov=True)
    # Solve equation for upper and lower bound
    solution_lower = shooting.solve_equation(solution_first, solution_second, solution_last, solution_second_last, grid,
                                             potential, left_bound,
                                             shooting.outer_turning_point(potential, left_bound, grid), numerov=True)
    solution_upper = shooting.solve_equation(solution_first, solution_second, solution_last, solution_second_last, grid,
                                             potential, right_bound,
                                             shooting.outer_turning_point(potential, right_bound, grid), numerov=True)
    print(shooting.generate_latex_table(eigenvalues, continuities, "$E/V_0$", "$F(E/V_0)$", rounding=5))
    return grid, [solution_lower, solution_upper, solution], \
        r"$\rho$", r"$\zeta(\rho)$", \
        [
         r"Solution for $\lambda = {}".format(left_bound),
         r"Solution for $\lambda = {}$".format(right_bound),
         r"Converged solution, $\lambda = {}$".format(round(eigenvalue, 4))
        ]


def excitons2c():
    def potential(x):
        return 0.25*x**2
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
    tolerance = 10**-8
    max_iterations = 100
    eigenvalue_guess = 0.4
    eigenvalues, continuity = shooting.shooting_method(grid, solution_first, solution_second, solution_last,
                                                       solution_second_last, tolerance, max_iterations, potential,
                                                       eigenvalue_guess, numerov=True, algorithm='improved')
    print(shooting.generate_latex_table(eigenvalues, continuity, "$E/V_0$", "$F(E/V_0)$", rounding=9))


@single_plot
def excitons3a():
    def potential(x):
        return 2/x**2 - 2/x
    # Set up equidistant grid
    grid_points = 1000
    grid_displacement = 0.01
    grid_end = 60
    grid = np.linspace(0, grid_end, grid_points) + grid_displacement
    # Guess an eigenvalue
    eigenvalue_guess = -0.06
    # Set up initial values for forward & backward solutions
    solution_first = grid[0]**2
    solution_second = grid[1]**2
    solution_last = np.exp(-np.sqrt(-eigenvalue_guess)*grid[-1])
    solution_second_last = np.exp(-np.sqrt(-eigenvalue_guess)*grid[-2])
    # Shoot!
    max_iterations = 100
    tolerance = 10**-12
    eigenvalues, derivative_continuities = shooting.shooting_method(grid, solution_first, solution_second,
                                                                    solution_last, solution_second_last,
                                                                    tolerance, max_iterations, potential,
                                                                    eigenvalue_guess, algorithm='improved',
                                                                    numerov=True)
    print(eigenvalues[-1], len(eigenvalues))
    # Solve equation
    turning_point = shooting.outer_turning_point(potential, eigenvalues[-1], grid)
    solution = shooting.solve_equation(solution_first, solution_second, solution_last, solution_second_last,
                                       grid, potential, eigenvalue_guess, turning_point, numerov=True)
    return grid, solution, r"$\rho$", r"$\zeta(\rho)$"

