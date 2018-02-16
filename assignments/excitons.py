import sys
import lib.excitons as excitons
import lib.util.util as util
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
        turning_points[n] = grid[abs((lambda x: 0.25*x**2)(grid) - energy).argmin()]

    return energies, turning_points, r"$\lambda$", "Turning point"


@plot_grid_show
def excitons1c():

    def potential(x):
        return 0.25*x**2

    # Eigenvalue guess
    eigenvalue_guess = 1.5

    # Set up equidistant grid
    grid_points = 100
    grid_displacement = 10**-5
    grid_end = 5
    grid = np.linspace(0, grid_end, grid_points) + grid_displacement

    # Calculate analytic solution
    analytic_solution = excitons.WaveFunction(grid, values=grid*np.exp(-grid**2/4))
    analytic_solution.normalize()

    # Set up initial values of forward and backward solutions
    boundary_left = (grid[0], grid[1])
    boundary_right = (analytic_solution[-2], analytic_solution[-1])

    # Solve equation
    solver = excitons.SchrodingerSolver(grid, potential, eigenvalue_guess, 0, boundary_left, boundary_right)
    solution = solver.solve(excitons.SchrodingerSolver.propagate_simple)[0]

    # Plot both solutions and analytic solution
    fig, ax = plt.subplots(2, sharex=True)
    solution.plot(ax[0], '.')
    analytic_solution.plot(ax[0])

    # Plot relative error
    error = (solution - analytic_solution).values
    ax[1].plot(grid, error)

    return ax


@vertical_subplots
def excitons2a():

    def potential(x):
        return 0.25*x**2

    # Eigenvalue guess
    eigenvalue_guess = 1.5

    # Set up equidistant grid
    grid_points = 100
    grid_displacement = 10**-5
    grid_end = 5
    grid = np.linspace(0, grid_end, grid_points) + grid_displacement

    # Calculate analytic solution
    analytic_solution = excitons.WaveFunction(grid, values=grid*np.exp(-grid**2/4))
    analytic_solution.normalize()

    # Set up initial values of forward and backward solutions
    boundary_left = (grid[0], grid[1])
    boundary_right = (analytic_solution[-2], analytic_solution[-1])

    # Solve the differential equation
    solver = excitons.SchrodingerSolver(grid, potential, eigenvalue_guess, 0, boundary_left, boundary_right)
    solution = solver.solve(excitons.SchrodingerSolver.propagate_numerov)[0]

    error = (solution - analytic_solution).values

    return grid, solution.values, error, r"$\rho", r"$\zeta(\rho)$", r"$\zeta(\rho) - \zeta_{00}(\rho)$"


@plot_single_window(r"$\rho$", r"$\zeta(\rho)$")
def excitons2b():

    def potential(x):
        return 0.25*x**2

    left_bound = 0.3
    right_bound = 1.8

    # Set up equidistant grid
    grid_points = 100
    grid_displacement = 10**-5
    grid_end = 5
    grid = np.linspace(0, grid_end, grid_points) + grid_displacement

    # Calculate analytic solution
    analytic_solution = excitons.WaveFunction(grid, values=grid*np.exp(-grid**2/4))
    analytic_solution.normalize()

    # Set up initial values of forward and backward solutions
    boundary_left = (grid[0], grid[1])
    boundary_right = (analytic_solution[-2], analytic_solution[-1])

    # Shoot!
    tolerance = 10**-4
    shooter = excitons.Shooter(grid, potential, boundary_left, boundary_right, 0)
    eigenvalues, derivative_continuities = shooter.shooter(tolerance, excitons.SchrodingerSolver.propagate_numerov,
                                                           shooter.bisection_iteration, left_bound, right_bound)
    eigenvalue = eigenvalues[-1]

    # Solve equation for found eigenvalue
    solution = shooter.get_solver(eigenvalue).solve(excitons.SchrodingerSolver.propagate_numerov)[0]

    # Solve equation for upper and lower bound
    solution_lower = shooter.get_solver(left_bound).solve(excitons.SchrodingerSolver.propagate_numerov)[0]
    solution_upper = shooter.get_solver(right_bound).solve(excitons.SchrodingerSolver.propagate_numerov)[0]

    print(util.generate_latex_table(eigenvalues, derivative_continuities, "$E/V_0$", "$F(E/V_0)$", rounding=5))

    return grid, [solution_lower.values, solution_upper.values, solution.values], \
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
    grid_displacement = 10**-5
    grid_end = 5
    grid = np.linspace(0, grid_end, grid_points) + grid_displacement

    # Calculate analytic solution
    analytic_solution = excitons.WaveFunction(grid, values=grid*np.exp(-grid**2/4))
    analytic_solution.normalize()

    # Set up initial values of forward and backward solutions
    boundary_left = (grid[0], grid[1])
    boundary_right = (analytic_solution[-2], analytic_solution[-1])

    # Shoot!
    tolerance = 10**-8
    eigenvalue_guess = 0.4
    shooter = excitons.Shooter(grid, potential, boundary_left, boundary_right, 0)
    eigenvalues, derivative_continuities = shooter.shooter(tolerance, excitons.SchrodingerSolver.propagate_numerov,
                                                           shooter.improved_iteration, eigenvalue_guess)

    print(util.generate_latex_table(eigenvalues, derivative_continuities, "$E/V_0$", "$F(E/V_0)$", rounding=9))


@plot_single_window(r"$\rho$", r"\zeta(\rho)")
def excitons3a():

    def potential(x):
        return -2/x
    angular_momentum = 1

    # Set up equidistant grid
    grid_points = 10000
    grid_displacement = 10**-5
    grid_end = 30
    grid = np.linspace(0, grid_end, grid_points) + grid_displacement

    # Guess an eigenvalue
    eigenvalue_guess = -0.25

    # Set up initial values for forward & backward solutions
    boundary_left = (grid[0]**2, grid[1]**2)
    boundary_right = (np.exp(-np.sqrt(-eigenvalue_guess)*grid[-2]), np.exp(-np.sqrt(-eigenvalue_guess)*grid[-1]))

    # Shoot!
    tolerance = 10**-8
    shooter = excitons.Shooter(grid, potential, boundary_left, boundary_right, angular_momentum)
    eigenvalues, derivative_continuities = shooter.shooter(tolerance, excitons.SchrodingerSolver.propagate_numerov,
                                                           shooter.improved_iteration, eigenvalue_guess)
    eigenvalue = eigenvalues[-1]

    # Solve equation
    solution = shooter.get_solver(eigenvalue).solve(excitons.SchrodingerSolver.propagate_numerov)[0]

    return grid, [solution.values], [r"$\lambda = {0}$".format(round(eigenvalue, 4))]


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
    boundary_left = (grid[0]**2, grid[1]**2)
    boundary_right = (np.exp(-np.sqrt(-eigenvalue_guess)*grid[-2]), np.exp(-np.sqrt(-eigenvalue_guess)*grid[-1]))

    # Shoot!
    tolerance = 10**-8

    shooter = excitons.Shooter(grid, potential, boundary_left, boundary_right, angular_momentum)
    eigenvalues, derivative_continuities = shooter.shooter(tolerance, excitons.SchrodingerSolver.propagate_numerov_log,
                                                           shooter.improved_iteration, eigenvalue_guess)
    eigenvalue = eigenvalues[-1]

    # Solve equation
    solution = shooter.get_solver(eigenvalue).solve(excitons.SchrodingerSolver.propagate_numerov_log)[0]

    return grid, [solution.values], [r"$\lambda = {0}$".format(round(eigenvalue, 4))]


@plot_single_window(r"$\rho$", r"\zeta(\rho)")
def excitons4a(potential=None):

    if potential is None:
        def potential(x):
            return -2/x

    # Set up equidistant grid, in log space
    # Then transform to a grid in \rho space
    grid_points = 8000
    grid_displacement = np.log(10**-4)
    grid_end = np.log(100)
    grid = np.linspace(grid_displacement, grid_end, grid_points)
    grid = np.exp(grid)

    # Set up initial values for forward & backward solutions
    boundary_left = (grid[0]**2, grid[1]**2)
    boundary_right = (np.exp(-grid[-2]), np.exp(-grid[-1]))

    # Iteration tolerance
    tolerance = 10**-8

    # Set up lists of quantum numbers
    principal_quantum_numbers = list(range(1, 6)) + [2, 3]
    angular_momenta = [0]*5 + [1, 2]

    # Calculate the eigenvalues by the shooting method
    # Using the exact eigenvalues as "Guesses"
    eigenvalues = []
    labels = []
    solutions = []
    for n, angular_momentum in zip(principal_quantum_numbers, angular_momenta):
        eigenvalue = -1/n**2
        print("Shooting with eigenvalue guess {0}".format(eigenvalue))
        shooter = excitons.Shooter(grid, potential, boundary_left, boundary_right, angular_momentum)
        _eigenvalues = shooter.shooter(tolerance, excitons.SchrodingerSolver.propagate_numerov_log,
                                       shooter.improved_iteration, eigenvalue)[0]
        eigenvalues.append(_eigenvalues[-1])
        print("Converged! Eigenvalue found: {0}".format(eigenvalues[-1]))

        # Some fun Pythonic goodness; generates list of labels to use in plotting
        labels.append(r"{0}{1}, $\lambda = {2}$".format(n, "s" if angular_momentum == 0
                                                        else "p" if angular_momentum == 1 else "d",
                                                        round(eigenvalues[-1], 4)))
        # Calculate solution from converged eigenvalue
        solutions.append(
                        shooter.get_solver(eigenvalues[-1])
                        .solve(excitons.SchrodingerSolver.propagate_numerov_log)[0].values
                        )

    # Print a table for in 4b
    print(util.generate_latex_table(
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

    # Stupid hack so I don't have to rewrite exercise 4a
    excitons4a(potential=haken)

