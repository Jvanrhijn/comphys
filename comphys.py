"""Executable for Computational Physics exercises

Command line args syntax:
    [assignment name in lowercase][exercise][subexercise]
    Example: comphys excitons1b
"""
import sys
import lib.shooting as shooting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
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
    pass


if __name__ == "__main__":
    eval(sys.argv[1]+"()")
