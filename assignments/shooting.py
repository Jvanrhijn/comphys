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
        turning_points[n] = grid[shooting.outer_turning_point_newton(lambda x: 0.5*x**2, energy, grid, 100)]
    ax.plot(energies, turning_points)
    ax.set_xlabel(r"\lambda")
    ax.set_ylabel("Turning point")
    ax.grid()
    plt.show()


def excitons1c():
    def potential(x):
        return 0.5*x**2
    # Eigenvalue guess
    eigenvalue_guess = 1.5
    # Set up equidistant grid
    grid_points = 100
    grid_displacement = 0.01
    grid_end = 4
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
                                                       grid, potential, eigenvalue_guess)
    solution_backward = shooting.solve_equation_backward(solution_backward_last, solution_backward_second_last,
                                                         grid, potential, eigenvalue_guess)
    # Match the solutions at the turning point
    solution_forward /= solution_forward[turning_point]
    solution_backward /= solution_backward[turning_point]
    # Glue solutions together
    solution = shooting.glue_arrays_together(solution_forward, solution_backward, turning_point)
    # Normalize solution
    solution = shooting.normalize_solution(grid, solution)
    # Plot both solutions and analytic solution
    fig, ax = plt.subplots(2, sharex='all')
    ax[0].plot(grid[0:turning_point+1], solution[0:turning_point+1], label="Forward solution")
    ax[0].plot(grid[turning_point:], solution[turning_point:], label="Backward solution")
    ax[1].set_xlabel(r"$\rho$")
    ax[0].set_ylabel(r"$\zeta(\rho)$")
    ax[0].plot(grid, analytic_solution, label="Analytic solution")
    ax[1].set_ylabel(r"$\zeta(\rho) - \zeta_{00}(\rho)$")
    # Plot relative error
    error = solution - analytic_solution
    ax[1].plot(grid, error)
    ax[0].grid(), ax[1].grid()
    ax[0].legend()
    plt.show()


if __name__ == "__main__":
    eval(sys.argv[1]+"()")

def excitons1c():
    def potential(x):
        return 0.5*x**2
    # Eigenvalue guess
    eigenvalue_guess = 1.5
    # Set up equidistant grid
    grid_points = 100
    grid_displacement = 0.01
    grid_end = 6
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
                                                       grid, potential, eigenvalue_guess)
    solution_backward = shooting.solve_equation_backward(solution_backward_last, solution_backward_second_last,
                                                         grid, potential, eigenvalue_guess)
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
    ax[1].set_ylabel(r"$\zeta(\rho) - \zeta_{00}(\rho)$")
    # Plot relative error
    error = solution - analytic_solution
    ax[1].plot(grid, error)
    ax[0].grid(), ax[1].grid()
    ax[0].legend()
    plt.show()

