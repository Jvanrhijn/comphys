import numpy as np
import matplotlib.pyplot as plt
import lib.wave_propagation as wp
from decorators.decorators import *
from tqdm import tqdm
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 16})
rc('text', usetex=True)


def potential_square(x, height, width):
    return height*(np.heaviside(x + 0.5*width, 1) - np.heaviside(x - 0.5*width, 1))


def potential_triangle(x, height, width, delta, reference: float=0):
    conditions = [x < -0.5*width, (x >= -0.5*width) & (x < 0.5*width), x >= 0.5*width]
    functions = [lambda y: reference,
                 lambda y: reference + height - delta/width*(y + 0.5*width),
                 lambda y: reference-delta]
    return np.piecewise(x, conditions, functions)


def transmission_square(energy, width):
    k = np.sqrt(energy)
    eta = np.sqrt(1 - energy)
    return (1 + ((k**2 + eta**2)/(2*k*eta)*np.sinh(eta*width))**2)**-1


def transmission_wkb(grid, energy, potential, prefactor=1):
    velocity_left = np.sqrt(energy - potential(grid[0]))
    velocity_right = np.sqrt(energy - potential(grid[-1]))
    return velocity_right/velocity_left*prefactor*np.exp(-2*np.trapz(np.sqrt(potential(grid[1:-1]) - energy),
                                                                     grid[1:-1]))


@plot_grid_show
def wave_propagation1a():
    grid = np.array([-1, -0.5, 0, 0.5])
    energies = np.linspace(0.01, 0.99, 100)
    transmissions = []
    for energy in tqdm(energies):
        transmission = wp.TransferMatrixSolver(grid, lambda x: potential_square(x, 1, 1), energy)\
            .calculate().transmission()[0]
        transmissions.append(transmission)
    fig, ax = plt.subplots(1)
    ax.plot(energies, transmissions, label="Numerical")
    ax.plot(energies[::5], transmission_square(energies[::5], 1), 'o', label="Analytical")
    ax.legend()
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$T$")
    return ax


@plot_grid_show
def wave_propagation1b(matrix_solver=wp.TransferMatrixSolver, energy=0.01):
    widths = np.linspace(0, 30, 100)
    transmissions_left = []
    transmissions_right = []
    for width in tqdm(widths):
        grid = np.array([-0.5*width-0.1, -0.5*width, 0, 0.5*width])
        tr_matrix = matrix_solver(grid, lambda x: potential_square(x, 1, width), energy).calculate()
        transmission_left, transmission_right = tr_matrix.transmission()
        transmissions_left.append(transmission_left), transmissions_right.append(transmission_right)
    fig, ax = plt.subplots(1)
    ax.semilogy(widths, transmissions_left, label="T")
    ax.semilogy(widths, transmissions_right, '--', label="T'")
    ax.set_xlabel(r"$a$"), ax.set_ylabel(r"$T$")
    ax.legend()
    return ax


def wave_propagation1c():
    wave_propagation1b(matrix_solver=wp.ScatterMatrixSolver)


def wave_propagation1d():
    wave_propagation1b(energy=0.5)


def wave_propagation2a():
    energy = 0.5
    grid = np.array([-1] + list(np.linspace(-0.5, 0.5, 150)))
    transmission_solver = wp.ScatterMatrixSolver(grid, lambda x: potential_triangle(x, 1, 1, 0.5), energy)
    transmission_solver.calculate()
    print("T = %f" % round(transmission_solver.transmission()[0], 5))


@plot_grid_show
def wave_propagation2b():
    grid = np.array([-1] + list(np.linspace(-0.5, 0.5, 150)))
    height, width, delta = 1, 1, 0.5
    energies = np.linspace(0.01, height-0.01, 100)
    transmission = []
    for energy in tqdm(energies):
        transmission_solver = wp.ScatterMatrixSolver(grid,
                                                     lambda x: potential_triangle(x, height, width, delta), energy)
        transmission_solver.calculate()
        transmission.append(transmission_solver.transmission()[0])
    fig, ax = plt.subplots(1)
    ax.plot(energies, transmission, label="Triangular barrier")
    ax.plot(energies, transmission_square(energies, width), label="Square barrier")
    ax.set_xlabel(r"$a$"), ax.set_ylabel(r"$T$")
    ax.legend()
    return ax


@plot_grid_show
def wave_propagation3a(delta=0.5, height=10):
    potential = lambda x: potential_triangle(x, height, width, delta)

    grid = np.array([-1] + list(np.linspace(-0.5, 0.5, 150)))
    width = 1
    energies = np.linspace(0.01, height-delta, 100)

    transmission = []
    wkb = []
    for energy in tqdm(energies):
        transmission_solver = wp.ScatterMatrixSolver(grid, potential, energy)
        transmission_solver.calculate()
        transmission.append(transmission_solver.transmission()[0])
        wkb_prefactor = 16*energy*(height - delta - energy)/height**2
        wkb.append(transmission_wkb(grid, energy, potential, wkb_prefactor))

    fig, ax = plt.subplots(1)
    ax.semilogy(energies, transmission, label="Scattering matrix")
    ax.semilogy(energies, wkb, label="WKB approximation")
    ax.legend()
    ax.set_xlabel(r"$a$"), ax.set_ylabel(r"$T$")
    return ax


def wave_propagation3b():
    wave_propagation3a(delta=2.5)


def wave_propagation3c():
    wave_propagation3a(delta=0.05, height=1)


@plot_grid_show
def wave_propagation4b(delta_builtin: float=0, potential_bias: np.ndarray=None):
    # Set up a potential bias grid; more points in the interesting (not flat) part of the curve
    if potential_bias is None:
        potential_bias = np.concatenate([np.linspace(-52.5, -26.5, 20),
                                         np.linspace(-26.5, 26.5, 10),
                                         np.linspace(26.5, 52.5, 20)])
    fermi_energy = 26.25
    height, width = 55.1, 3
    grid = np.array([-2] + list(np.linspace(-1.5, 1.5, 150)))
    currents = []
    for delta in tqdm(potential_bias):
        delta = delta + delta_builtin
        potential = lambda x: potential_triangle(x, height, width, delta, reference=0)

        # Energy grid; include small offset in lower bound to prevent division by zero
        lower_integral_bound = fermi_energy - delta if fermi_energy > delta else 0
        energies = np.linspace(lower_integral_bound + 0.01, fermi_energy, 100)

        transmission = np.zeros(len(energies))
        for jj, energy in enumerate(energies):
            transmission_solver = wp.ScatterMatrixSolver(grid, potential, energy)
            transmission_solver.calculate().transmission()
            transmission[jj] = transmission_solver.transmission()[0]
        currents.append(-np.trapz(transmission, energies))

    currents = np.array(currents)
    fig, ax = plt.subplots(1)
    # Constants for converting scaled units to volts and amperes
    volts = -38.1*10**-3  # eV; set e = 1 to get voltage, minus sign for electron charge
    amperes = 2.952*10**-6
    ax.plot(potential_bias*volts, currents*amperes*10**9)
    ax.set_xlabel(r"$U$ (V)"), ax.set_ylabel(r"$I_T$ (nA)")
    return ax


def wave_propagation4c():
    potential_bias = np.concatenate([np.linspace(-52.5, 0, 10),
                                     np.linspace(0, 52.5, 40)])
    wave_propagation4b(delta_builtin=36.7, potential_bias=potential_bias)


@plot_grid_show
def wave_propagation4d():
    energy_grid_length = 100

    potential_bias = np.concatenate([np.linspace(-52.5, 0, 10),
                                     np.linspace(0, 12.25, 20),
                                     np.linspace(12.25, 52.5, 40)])
    fermi_energy = 26.25
    height, width = 55.1, 3
    delta_builtin = 36.7
    grid = np.array([-2] + list(np.linspace(-1.5, 1.5, 150)))
    currents = []

    def calculate_transmission(grid, potential, energy_grid):
        transmission_ = np.zeros(len(energy_grid))
        for jj, energy in enumerate(energy_grid):
            transmission_solver = wp.ScatterMatrixSolver(grid, potential, energy)
            transmission_solver.calculate()
            transmission_[jj] = abs(transmission_solver.transmission()[0])
        return transmission_

    for delta in tqdm(potential_bias):
        delta = delta + delta_builtin
        potential = lambda x: potential_triangle(x, height, width, delta, reference=0)

        if delta > 0:
            lower_bound = 0
            upper_bound = fermi_energy
            energy_grid = np.linspace(lower_bound + 0.01, upper_bound, energy_grid_length)
            transmission = calculate_transmission(grid, potential, energy_grid)
            index = abs(energy_grid).argmin()
            currents.append(
                -delta*np.trapz(transmission[:index], energy_grid[:index])
                - np.trapz((fermi_energy - energy_grid[index:])*transmission[index:], energy_grid[index:])
            )
        else:
            lower_bound = -delta
            upper_bound = fermi_energy-delta
            energy_grid = np.linspace(lower_bound + 0.01, upper_bound, energy_grid_length)
            transmission = calculate_transmission(grid, potential, energy_grid)
            index = abs(energy_grid - fermi_energy).argmin()
            currents.append(
                -delta*np.trapz(transmission[:index], energy_grid[:index])
                - np.trapz((fermi_energy - delta - energy_grid[index:])*transmission[index:], energy_grid[index:])
            )

    currents = np.array(currents)
    fig, ax = plt.subplots(1)
    # Constants for converting scaled units to volts and amperes
    volts = -38.1*10**-3  # eV; set e = 1 to get voltage, minus sign for electron charge
    amperes = 0.47
    ax.plot(potential_bias*volts, currents*amperes*10**3)
    ax.set_xlabel(r"$U$ (V)"), ax.set_ylabel(r"$I_T$ (mA)")
    return ax
