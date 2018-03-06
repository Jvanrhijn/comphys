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
            ax[i, j].plot(mc_paramagnet.magnetizations/lattice_side**2)
            ax[i, j].grid('on')
            mc_paramagnet.reset()
    plt.show()


def monte_carlo_1_5d():
    lattice_side, field, num_runs = 10, 0.5, 5000
    mc_paramagnet = monte_carlo.ParaMagnet(num_runs, field, lattice_side)
    mc_paramagnet.set_equilibration_time(200)
    mc_paramagnet.simulate()
    magnetization, m_stdev = mc_paramagnet.mean_magnetization()
    print("m(S) = {m:.{dig_m}f} +/- {s:.{dig_s}f}".format(m=magnetization, dig_m=3, s=m_stdev, dig_s=3))


def monte_carlo_1_5e():
    lattice_sides = [5, 15, 30]
    num_runs = 5000
    field = 0.5
    mc_paramagnet = monte_carlo.ParaMagnet(num_runs, field, 0)
    mc_paramagnet.set_equilibration_time(500)
    fig, ax = plt.subplots(1, len(lattice_sides), sharey=True)
    for n, lattice_side in enumerate(lattice_sides):
        mc_paramagnet.set_lattice_side(lattice_side)
        mc_paramagnet.reset()
        mc_paramagnet.simulate()
        ax[n].plot(mc_paramagnet.magnetizations / lattice_side**2)
        ax[n].grid('on')
        ax[n].set_title(r'$L = %i$' % lattice_side)
    plt.show()


def monte_carlo_1_6b():
    lattice_side = 10
    num_runs = 50
    field = 0.5
    mc_paramagnet = monte_carlo.ParaMagnet(num_runs, field, lattice_side)
    mc_paramagnet.set_equilibration_time(10)
    mc_paramagnet.simulate_unit()
    mc_paramagnet.plot_results()
    plt.show()


def monte_carlo_1_7():
    field_strengths = np.linspace(-2, 2, 20)
    num_runs = 100
    lattice_side = 10
    mc_paramagnet = monte_carlo.ParaMagnet(num_runs, 0, lattice_side)
    mc_paramagnet.set_equilibration_time(5)
    magnetization_mean = []
    magnetization_stdev = []
    for field in field_strengths:
        mc_paramagnet.set_magnetic_field(field)
        mc_paramagnet.reset()
        mc_paramagnet.simulate_unit()
        m_mean, m_stdev = mc_paramagnet.mean_magnetization()
        magnetization_mean.append(m_mean)
        magnetization_stdev.append(m_stdev)

    fig, ax = plt.subplots(1)
    ax.errorbar(field_strengths, magnetization_mean, yerr=magnetization_stdev, fmt='o', label='Monte Carlo result')
    ax.grid('on')
    ax.set_xlabel(r'$B$'), ax.set_ylabel(r'$\langle m \rangle$')
    ax.plot(field_strengths, np.tanh(field_strengths), label='Exact')
    ax.legend()
    plt.show()


def monte_carlo_1_8a():
    field_strengths = np.linspace(-2, 2, 50)
    num_runs = 5000
    lattice_side = 10
    exact_susceptibility = 1/np.cosh(field_strengths)**2
    mc_paramagnet = monte_carlo.ParaMagnet(num_runs, 0, lattice_side)
    mc_paramagnet.set_equilibration_time(5)
    susceptibility = []
    for field in field_strengths:
        mc_paramagnet.set_magnetic_field(field)
        mc_paramagnet.reset()
        mc_paramagnet.simulate_unit()
        susceptibility.append(mc_paramagnet.susceptibility())
    fig, ax = plt.subplots(1)
    ax.plot(field_strengths, susceptibility, 'o', label='Monte Carlo result')
    ax.plot(field_strengths, exact_susceptibility, label='Exact')
    ax.grid('on')
    ax.set_xlabel(r"$B$")
    ax.set_ylabel(r"$\chi/N$")
    ax.legend()
    plt.show()


def monte_carlo_1_8b():
    field = 1.
    field_shifts = np.linspace(0.001, 0.15, 100)
    lattice_side = 10
    num_runs = 5000
    mc_paramagnet = monte_carlo.ParaMagnet(num_runs, field, lattice_side)
    susceptibility_list = []

    def susceptibility_finite_difference(shift, mc_solver: monte_carlo.ParaMagnet):
        mc_solver.set_magnetic_field(field + shift/2)
        mc_solver.reset()
        mc_solver.simulate_unit()
        magnetization_upper = mc_solver.mean_magnetization()[0]
        mc_solver.set_magnetic_field(field - shift/2)
        mc_solver.reset()
        mc_solver.simulate_unit()
        magnetization_lower = mc_solver.mean_magnetization()[0]
        return (magnetization_upper - magnetization_lower)/shift

    for field_shift in field_shifts:
        susceptibility = susceptibility_finite_difference(field_shift, mc_paramagnet)
        susceptibility_list.append(susceptibility)

    fig, ax = plt.subplots(1)
    ax.plot(field_shifts, np.ones(len(field_shifts))*1/np.cosh(1)**2, label=r'$1/\cosh^2(1)$')
    ax.plot(field_shifts, susceptibility_list, 'o', label=r'$\Delta\chi / \Delta B$')
    ax.set_xlabel(r"$\Delta B$")
    ax.set_ylabel(r"$\partial (\chi/N) / \partial B$")
    ax.legend()
    ax.grid('on')
    plt.show()


def monte_carlo_2_3a():
    pass
