import sys
import lib.shooting as shooting
import numpy as np
from decorators.decorators import *
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 12})
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

    # Solve the differential equation, including matching forward and backward solutions and normalization
    solution = shooting.solve_equation(solution_forward_first, solution_forward_second, solution_backward_last,
                                       solution_backward_second_last, grid, potential, eigenvalue_guess, turning_point,
                                       shooting.solution_next)

    # Plot both solutions and analytic solution
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(grid, analytic_solution, label="Analytic solution")
    ax[0].plot(grid[0:turning_point+1], solution[0:turning_point+1], 'x', label="Forward solution")
    ax[0].plot(grid[turning_point:], solution[turning_point:], 'x', label="Backward solution")
    ax[1].set_xlabel(r"$\rho$")
    ax[0].set_ylabel(r"$\zeta(\rho)$")
    ax[1].set_ylabel(r"$\zeta(\rho) - \zeta_{00}(\rho)$")

    # Plot relative error
    error = solution - analytic_solution
    ax[1].plot(grid, error)
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
                                               turning_point, shooting.solution_next_numerov)
    error_numerov = solution_numerov - analytic_solution

    return grid, solution_numerov, error_numerov, r"$\rho", r"$\zeta(\rho)", r"$\zeta(\rho) - \zeta_{00}(\rho)$"


@plot_single_window(r"$\rho$", r"$\zeta(\rho)$")
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

    # Shoot!
    tolerance = 10**-4
    max_iterations = 100
    eigenvalues, continuities, = shooting.shooting_method(grid, solution_first, solution_second, solution_last,
                                                          solution_second_last, tolerance, max_iterations, potential,
                                                          shooting.solution_next_numerov,
                                                          left_bound, right_bound
                                                          )
    eigenvalue = eigenvalues[-1]

    # Get classical turning point index
    turning_point = shooting.outer_turning_point(potential, eigenvalue, grid)

    # Solve equation for found eigenvalue
    solution = shooting.solve_equation(solution_first, solution_second, solution_last, solution_second_last, grid,
                                       potential, eigenvalue, turning_point, shooting.solution_next_numerov)

    # Solve equation for upper and lower bound
    solution_lower = shooting.solve_equation(solution_first, solution_second, solution_last, solution_second_last, grid,
                                             potential, left_bound,
                                             shooting.outer_turning_point(potential, left_bound, grid),
                                             shooting.solution_next_numerov)
    solution_upper = shooting.solve_equation(solution_first, solution_second, solution_last, solution_second_last, grid,
                                             potential, right_bound,
                                             shooting.outer_turning_point(potential, right_bound, grid),
                                             shooting.solution_next_numerov)

    print(shooting.generate_latex_table(eigenvalues, continuities, "$E/V_0$", "$F(E/V_0)$", rounding=5))

    return grid, [solution_lower, solution_upper, solution], \
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

    # Shoot!
    tolerance = 10**-8
    max_iterations = 100
    eigenvalue_guess = 0.4
    eigenvalues, continuity = shooting.shooting_method(grid, solution_first, solution_second, solution_last,
                                                       solution_second_last, tolerance, max_iterations, potential,
                                                       shooting.solution_next_numerov, eigenvalue_guess,
                                                       algorithm='improved')

    print(shooting.generate_latex_table(eigenvalues, continuity, "$E/V_0$", "$F(E/V_0)$", rounding=9))


@plot_single_window(r"$\rho$", r"\zeta(\rho)")
def excitons3a():

    def potential(x):
        return -2/x + 2/x**2

    # Set up equidistant grid
    grid_points = 10000
    grid_displacement = 10**-5
    grid_end = 30
    grid = np.linspace(0, grid_end, grid_points) + grid_displacement

    # Guess an eigenvalue
    eigenvalue_guess = -0.25

    # Set up initial values for forward & backward solutions
    solution_first = grid[0]**2
    solution_second = grid[1]**2
    solution_last = np.exp(-np.sqrt(-eigenvalue_guess)*grid[-1])
    solution_second_last = np.exp(-np.sqrt(-eigenvalue_guess)*grid[-2])

    # Shoot!
    max_iterations = 100
    tolerance = 10**-10
    eigenvalues, derivative_continuities = shooting.shooting_method(
                                                                    grid, solution_first, solution_second,
                                                                    solution_last, solution_second_last,
                                                                    tolerance, max_iterations, potential,
                                                                    shooting.solution_next_numerov,
                                                                    eigenvalue_guess, algorithm='improved',
                                                                    )
    eigenvalue = eigenvalues[-1]

    # Solve equation
    turning_point = shooting.outer_turning_point(potential, eigenvalue, grid)
    solution = shooting.solve_equation(solution_first, solution_second, solution_last, solution_second_last,
                                       grid, potential, eigenvalue, turning_point, shooting.solution_next_numerov)

    return grid, [solution], [r"$\lambda = {0}$".format(round(eigenvalue, 4))]


@plot_single_window(r"$\rho$", r"$\zeta(\rho)$")
def excitons3b():

    def potential(x):
        return -2/x
    angular_momentum = 1

    # Set up equidistant grid, in log space
    # Then transform to a grid in \rho space
    grid_points = 10000
    grid_displacement = np.log(10**-5)
    grid_end = np.log(30)
    grid = np.linspace(grid_displacement, grid_end, grid_points)
    grid = np.exp(grid)

    # Guess an eigenvalue
    eigenvalue_guess = -0.4

    # Set up initial values for forward & backward solutions
    solution_first = grid[0]**2
    solution_second = grid[1]**2
    solution_last = np.exp(-grid[-1])
    solution_second_last = np.exp(-grid[-2])

    # Shoot!
    max_iterations = 100
    tolerance = 10**-8
    eigenvalues, derivative_continuities = shooting.shooting_method(
                                                                    grid, solution_first, solution_second,
                                                                    solution_last, solution_second_last,
                                                                    tolerance, max_iterations, potential,
                                                                    lambda *args: shooting.solution_next_log_numerov(
                                                                        *args, angular_momentum=angular_momentum
                                                                    ),
                                                                    eigenvalue_guess, algorithm='improved',
                                                                    )
    eigenvalue = eigenvalues[-1]

    # Solve equation
    turning_point = shooting.outer_turning_point(potential, eigenvalue, grid)
    solution = shooting.solve_equation(solution_first, solution_second, solution_last, solution_second_last,
                                       grid, potential, eigenvalue, turning_point,
                                       lambda *args: shooting.solution_next_log_numerov(
                                           *args, angular_momentum=angular_momentum))

    return grid, [solution], [r"$\lambda = {0}$".format(round(eigenvalue, 4))]


@plot_single_window(r"$\rho$", r"\zeta(\rho)")
def excitons4a(potential=None):

    if potential is None:
        def potential(x):
            return -2/x

    # Set up equidistant grid, in log space
    # Then transform to a grid in \rho space
    grid_points = 10000
    grid_displacement = np.log(10**-4)
    grid_end = np.log(100)
    grid = np.linspace(grid_displacement, grid_end, grid_points)
    grid = np.exp(grid)

    # Set up initial values for forward & backward solutions
    solution_first = grid[0]**2
    solution_second = grid[1]**2
    solution_last = np.exp(-grid[-1])
    solution_second_last = np.exp(-grid[-2])

    # Iteration parameters
    tolerance = 10**-8
    max_iterations = 100

    # Set up lists of quantum numbers
    principal_quantum_numbers = list(range(1, 6)) + [2, 3]
    angular_momenta = [0]*5 + [1, 2]

    # Calculate the eigenvalues by the shooting method
    # Using the exact eigenvalues as "Guesses"
    eigenvalues = []
    labels = []
    for n, angular_momentum in zip(principal_quantum_numbers, angular_momenta):
        eigenvalue = -1/n**2
        print("Shooting with eigenvalue guess {0}".format(eigenvalue))
        _eigenvalues = shooting.shooting_method(
                                                grid, solution_first, solution_second,
                                                solution_last, solution_second_last,
                                                tolerance, max_iterations, potential,
                                                lambda *args:
                                                shooting.solution_next_log_numerov(
                                                 *args, angular_momentum=angular_momentum
                                                ),
                                                eigenvalue, algorithm='improved'
                                                )[0]
        eigenvalues.append(_eigenvalues[-1])
        print("Converged! Eigenvalue found: {0}".format(eigenvalues[-1]))

        # Some fun Pythonic goodness; generates list of labels to use in plotting
        labels.append(r"{0}{1}, $\lambda = {2}$".format(n, "s" if angular_momentum == 0
                                                        else "p" if angular_momentum == 1 else "d",
                                                        round(eigenvalues[-1], 4)))

    # Calculate solutions from obtained eigenvalues
    turning_points = [shooting.outer_turning_point(potential, eigenvalue, grid) for eigenvalue in eigenvalues]
    solutions = [
                    shooting.solve_equation(solution_first, solution_second, solution_last, solution_second_last,
                                            grid, potential, eigenvalue, turning_point,
                                            lambda *args: shooting.solution_next_log_numerov(
                                             *args, angular_momentum=angular_momentum))
                    for eigenvalue, turning_point, angular_momentum in zip(eigenvalues, turning_points, angular_momenta)
                ]

    # Print a table for in 4b
    print(shooting.generate_latex_table(
        [label[:2] for label in labels],
        [eigenvalue * 0.2748744547718782 + 2.17202 for eigenvalue in eigenvalues],
        "State", "Energy"))

    return grid, solutions, labels


def excitons4b():
    # Enter fundamental constants
    hbar = 1.0545718 * 10**-34  # Js
    electric_constant = 8.85418782 * 10**-12  # m**-3 kg**-1 s**4 A**2
    electron_mass = 9.10938356 * 10**-31  # kg
    dielectric_constant = 7.5 * electric_constant
    electron_charge = 1.60217662 * 10**-19
    reduced_mass = 0.99*0.57/(0.99 + 0.57)*electron_mass
    characteristic_radius = 4*np.pi*dielectric_constant*hbar**2/(reduced_mass*electron_charge**2)

    print("Characteristic radius = {} nm".format(characteristic_radius*10**9))

    Rydberg = reduced_mass*electron_charge**2/(32*np.pi*dielectric_constant**2*hbar**2)


@plot_single_window(r"$\rho$", r"$V(\rho)/V_0$")
def excitons4c():

    def coulomb(x):
        return -2/x

    electric_constant = 8.85418782 * 10**-12  # m**-3 kg**-1 s**4 A**2
    eps = 7.5 * electric_constant
    eps_1 = 7.11 * eps
    eps_2 = 6.45 * eps
    characteristic_radius = 1.0971776306592957
    rho_p1 = 3.573/characteristic_radius
    rho_m1 = 2.711/characteristic_radius
    rho_p2 = 1.656/characteristic_radius
    rho_m2 = 1.257/characteristic_radius

    def haken(x):
        return -2/x * (1 - 0.5*(np.e**(-x/rho_p1) + np.e**(-x/rho_m1)) + eps/eps_1 *
                       (0.5*(np.e**(-x/rho_p1) + np.e**(-x/rho_m1)) - 0.5*(np.e**(-x/rho_p2) + np.e**(-x/rho_m2)))
                       + eps/eps_2 * 0.5*(np.e**(-x/rho_p2) + np.e**(-x/rho_m2)))

    grid = np.linspace(0.5, 8, 1000)
    labels = ["Coulomb potential", "Haken potential"]

    return grid, [coulomb(grid), haken(grid)], labels


def excitons4d():

    # Haken potential; the magic numbers are calculated from the given constants
    # Improves speed versus defining them or calculating them in the function body
    def haken(x):
        return -2/x * (1
                       - 0.5*(np.e**(-x/3.256537410312475) + np.e**(-x/2.4708852279197084))
                       + 0.14064697609001403
                       * (0.5*(np.e**(-x/3.256537410312475) + np.e**(-x/2.4708852279197084))
                          - 0.5*(np.e**(-x/1.5093271624622049) + np.e**(-x/1.145666813535623)))
                       + 0.1550387596899225
                       * 0.5*(np.e**(-x/1.5093271624622049) + np.e**(-x/1.145666813535623)))

    excitons4a(potential=haken)

