import sys
import lib.shooting as shooting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


def excitons1b():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    energies = np.linspace(0, 10, 1000)
    grid = np.linspace(0, 10, 1000)
    turning_points = np.zeros(len(grid))
    for n, energy in enumerate(energies):
        turning_points[n] = grid[shooting.outer_turning_point_newton(lambda x: 0.25*x**2, energy, grid, 100)]
    ax.plot(energies, turning_points)
    ax.set_xlabel(r"\lambda")
    ax.set_ylabel("Turning point")
    ax.grid()
    plt.show()


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
    turning_point = shooting.outer_turning_point_newton(potential, eigenvalue_guess, grid, len(grid)//2)
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
    ax[0].plot(grid[0:turning_point+1], solution[0:turning_point+1], label="Forward solution")
    ax[0].plot(grid[turning_point:], solution[turning_point:], label="Backward solution")
    ax[1].set_xlabel(r"$\rho$")
    ax[0].set_ylabel(r"$\zeta(\rho)$")
    ax[0].plot(grid, analytic_solution, label="Analytic solution")
    ax[1].set_ylabel(r"$\zeta(\rho) - \zeta_{00}(\rho)$, $10^{-4}$")
    # Plot relative error
    error = solution - analytic_solution
    ax[1].plot(grid, error*10**4)
    ax[0].grid(), ax[1].grid()
    ax[0].legend()
    plt.show()


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
    turning_point = shooting.outer_turning_point_newton(potential, eigenvalue_guess, grid, len(grid)//2)
    # Set up initial values of forward and backward solutions
    solution_first = grid[0]
    solution_second = grid[1]
    solution_last = analytic_solution[-1]
    solution_second_last = analytic_solution[-2]
    # Solve the differential equation
    solution_numerov = shooting.solve_equation(solution_first, solution_second, solution_last,
                                               solution_second_last, grid, potential, eigenvalue_guess,
                                               turning_point, numerov=True)
    solution_naive = shooting.solve_equation(solution_first, solution_second, solution_last,
                                               solution_second_last, grid, potential, eigenvalue_guess,
                                               turning_point, numerov=False)
    # Plot both solutions and analytic solution
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(grid, solution_numerov, label="Numerov")
    ax[1].set_xlabel(r"$\rho$")
    ax[0].set_ylabel(r"$\zeta(\rho)$")
    ax[0].plot(grid, analytic_solution, label="Analytic solution")
    ax[1].set_ylabel(r"$\zeta(\rho) - \zeta_{00}(\rho)$, $10^4$")
    # Plot relative error
    error_numerov = solution_numerov - analytic_solution
    error_naive = solution_naive - analytic_solution
    ax[1].plot(grid, error_numerov*10**4)
    ax[0].grid(), ax[1].grid()
    ax[0].legend()
    plt.show()
