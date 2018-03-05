import matplotlib.pyplot as plt
import numpy as np
import lib.monte_carlo as monte_carlo
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 12})
rc('text', usetex=True)


def monte_carlo_1_5a():
    lattice_side = 10
    field = 0.5
    num_runs = 2000
    mc_paramagnet = monte_carlo.ParaMagnet(num_runs, field, lattice_side)
    mc_paramagnet.set_equilibration_time(200)
    fig, ax = plt.subplots(2, 2)
    # Run simulation four times, plot each
    for i in range(0, 2):
        for j in range(0, 2):
            mc_paramagnet.simulate()
            ax[i, j].plot(mc_paramagnet.magnetizations)
            ax[i, j].grid('on')
            mc_paramagnet.reset()
    plt.show()
