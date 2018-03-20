import numpy as np
import matplotlib.pyplot as plt
import lib.wave_propagation as wp
from decorators.decorators import *
from tqdm import tqdm
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 12})
rc('text', usetex=True)


def potential_square(x, height, width):
    return height*(np.heaviside(x + 0.5*width, 1) - np.heaviside(x - 0.5*width, 1))


def transmission_square(energy, width):
    k = np.sqrt(energy)
    eta = np.sqrt(1 - energy)
    return (1 + ((k**2 + eta**2)/(2*k*eta)*np.sinh(eta*width))**2)**-1


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
def wave_propagation1b(matrix_solver=wp.TransferMatrixSolver):
    widths = np.linspace(0, 30, 100)
    energy = 0.01
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

