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
    figures = []
    axes = []
    # Run simulation four times, plot each
    for n in range(0, 4):
        mc_paramagnet.simulate()
        mc_paramagnet.plot_results()
        fig, ax = mc_paramagnet.plot_results()
        fig.suptitle(r"Run #{}".format(n+1))
        mc_paramagnet.reset()
    plt.show()
