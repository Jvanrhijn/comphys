import matplotlib.pyplot as plt
import numpy as np
import lib.monte_carlo as monte_carlo
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 12})
rc('text', usetex=True)


def monte_carlo_1_5a():
    lattice_side = 10
    field = 0.5
    num_runs = 200000
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


def monte_carlo_1_5d():
    lattice_side, field, num_runs = 10, 0.5, 2000
    mc_paramagnet = monte_carlo.ParaMagnet(num_runs, field, lattice_side)
    mc_paramagnet.set_equilibration_time(200)
    mc_paramagnet.simulate()
    magnetization, m_stdev = mc_paramagnet.mean_magnetization()
    print("m(S) = {m:.{dig_m}f} +/- {s:.{dig_s}f}".format(m=magnetization, dig_m=3, s=m_stdev, dig_s=3))


def monte_carlo_1_5e():
    lattice_sides = [5, 10, 15]
    num_runs = 2000
    field = 0.5
    mc_paramagnet = monte_carlo.ParaMagnet(num_runs, field, 0)
    mc_paramagnet.set_equilibration_time(200)
    fig, ax = plt.subplots(1, 3, sharey=True)
    for n, lattice_side in enumerate(lattice_sides):
        mc_paramagnet.set_lattice_side(lattice_side)
        mc_paramagnet.simulate()
        ax[n].plot(mc_paramagnet.magnetizations / lattice_side**2)
        ax[n].grid('on')
        ax[n].set_title(r'$L = %i$' % lattice_side)
        mc_paramagnet.reset()
    plt.show()


def monte_carlo_1_6b():
    lattice_side = 10
    num_runs = 2000
    field = 0.5
    mc_paramagnet = monte_carlo.ParaMagnet(num_runs, field, lattice_side, unit_step=True)
    mc_paramagnet.set_equilibration_time(1)
    mc_paramagnet.simulate()
    mc_paramagnet.plot_results()
    plt.show()

