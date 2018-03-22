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
    grid = np.array([-1] + list(np.linspace(-0.5, 0.5, 13000)))
    transmission_solver = wp.ScatterMatrixSolver(grid, lambda x: potential_triangle(x, 1, 1, 0.5), energy)
    transmission_solver.calculate()
    print("T = %f" % round(transmission_solver.transmission()[0], 5))


@plot_grid_show
def wave_propagation2b():
    grid = np.array([-1] + list(np.linspace(-0.5, 0.5, 13000)))
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
    ax.set_xlabel(r"$\lambda$"), ax.set_ylabel(r"$T$")
    ax.legend()
    return ax


@plot_grid_show
def wave_propagation3a(delta=0.5, height=10):
    potential = lambda x: potential_triangle(x, height, width, delta)

    grid = np.array([-1] + list(np.linspace(-0.5, 0.5, 13000)))
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
    ax.set_xlabel(r"$\lambda$"), ax.set_ylabel(r"$T$")
    return ax


def wave_propagation3b():
    wave_propagation3a(delta=2.5)


def wave_propagation3c():
    wave_propagation3a(delta=0.05, height=1)


@plot_grid_show
def wave_propagation4b(delta_builtin=0):
    potential_bias = np.linspace(-52.5, 52.5, 20)
    fermi_energy = 26.24684
    height, width = 55.1, 3
    grid = np.array([-2] + list(np.linspace(-1.5, 1.5, 100)))
    currents = []
    for delta in tqdm(potential_bias):
        delta = delta + delta_builtin
        potential = lambda x: potential_triangle(x, height, width, delta, reference=0)

        lower_integral_bound = fermi_energy - delta if fermi_energy > delta else 0
        energies = np.linspace(lower_integral_bound, fermi_energy, 100)

        transmission = np.zeros(len(energies))
        for jj, energy in enumerate(energies):
            transmission_solver = wp.ScatterMatrixSolver(grid, potential, energy)
            transmission_solver.calculate().transmission()
            transmission[jj] = transmission_solver.transmission()[0]
        currents.append(-np.trapz(transmission, energies))

    currents = np.array(currents)
    fig, ax = plt.subplots(1)
    ax.plot(potential_bias, currents, '.')
    ax.set_xlabel(r"$\Delta \phi$ (V)"), ax.set_ylabel(r"$I_T$ (A)")
    return ax

