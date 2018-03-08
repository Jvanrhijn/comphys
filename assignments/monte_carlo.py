import itertools
import matplotlib.pyplot as plt
import numpy as np
import lib.monte_carlo as monte_carlo
from tqdm import tqdm
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 12})
rc('text', usetex=True)


def monte_carlo_1_5a():
    lattice_side = 10
    field = 0.5
    num_runs = 2000
    mc_paramagnet = monte_carlo.ParaMagnet(num_runs, field, lattice_side)
    mc_paramagnet.equilibration_time = 200
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
    mc_paramagnet.equilibration_time = 200
    mc_paramagnet.simulate()
    magnetization, m_stdev = mc_paramagnet.mean_magnetization()
    print("m(S) = {m:.{dig_m}f} +/- {s:.{dig_s}f}".format(m=magnetization, dig_m=3, s=m_stdev, dig_s=3))


def monte_carlo_1_5e():
    lattice_sides = [5, 15, 30]
    num_runs = 5000
    field = 0.5
    mc_paramagnet = monte_carlo.ParaMagnet(num_runs, field, 0)
    mc_paramagnet.equilibration_time = 500
    fig, ax = plt.subplots(1, len(lattice_sides), sharey=True)
    for n, lattice_side in enumerate(lattice_sides):
        mc_paramagnet.lattice_side = lattice_side
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
    mc_paramagnet.equilibration_time = 10
    mc_paramagnet.simulate_unit()
    mc_paramagnet.plot_results()
    plt.show()


def monte_carlo_1_7():
    field_strengths = np.linspace(-2, 2, 20)
    num_runs = 100
    lattice_side = 10
    mc_paramagnet = monte_carlo.ParaMagnet(num_runs, 0, lattice_side)
    mc_paramagnet.equilibration_time = 5
    magnetization_mean = []
    magnetization_stdev = []
    for field in field_strengths:
        mc_paramagnet.magnetic_field = field
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
    mc_paramagnet.equilibration_time = 5
    susceptibility = []
    for field in field_strengths:
        mc_paramagnet.magnetic_field = field
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
        mc_solver.magnetic_field = field + shift/2
        mc_solver.simulate_unit()
        magnetization_upper = mc_solver.mean_magnetization()[0]
        mc_solver.magnetic_field = field - shift/2
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
    field, lattice_side, coupling, num_runs = 0, 10, 1, 200
    # i, ii
    mc_ferromagnet = monte_carlo.FerroMagnet(num_runs, field, coupling, lattice_side)
    mc_ferromagnet.equilibration_time = 20
    mc_ferromagnet.simulate_unit()
    fig_no_field = mc_ferromagnet.plot_results()[0]
    fig_no_field.suptitle(r"$B = 0$")
    # iii
    mc_ferromagnet.magnetic_field = 0.05
    mc_ferromagnet.simulate_unit()
    fig_weak_field = mc_ferromagnet.plot_results()[0]
    fig_weak_field.suptitle(r"$B = 0.05$")
    mc_ferromagnet.magnetic_field = 1
    mc_ferromagnet.simulate_unit()
    fig_strong_field = mc_ferromagnet.plot_results()[0]
    fig_strong_field.suptitle(r"$B = 1$")
    plt.show()


def monte_carlo_2_3b():
    field, lattice_side, num_runs, coupling = 0, 10, 200, 0.2
    mc_ferromagnet = monte_carlo.FerroMagnet(num_runs, field, coupling, lattice_side)
    mc_ferromagnet.equilibration_time = 20
    mc_ferromagnet.simulate_unit()
    mc_ferromagnet.plot_results()
    plt.show()


def monte_carlo_2_4ab():
    num_runs = 1000
    lattice_side = 10
    couplings = np.linspace(0, 1, 20)
    mc_ferromagnet = monte_carlo.FerroMagnet(num_runs, 0, 0, lattice_side)
    mc_ferromagnet.equilibration_time = 100
    fig, ax = plt.subplots(1, 2)
    for axis in ax:
        axis.set_xlabel(r"$J$")
        axis.grid("on")
    ax[0].set_ylabel(r"$m$")
    ax[1].set_ylabel(r"$\chi$")

    for coupling in couplings:
        magnetizations = []
        susceptibilities = []
        mc_ferromagnet.coupling = coupling
        for _ in range(5):
            mc_ferromagnet.reset()
            mc_ferromagnet.simulate_unit()
            magnetization = mc_ferromagnet.mean_magnetization()[0]
            susceptibility = mc_ferromagnet.susceptibility()
            magnetizations.append(magnetization)
            susceptibilities.append(susceptibility)
        ax[0].plot(np.ones(len(magnetizations))*coupling, magnetizations, '.', color='#1f77b4')
        ax[1].plot(np.ones(len(susceptibilities))*coupling, susceptibilities, '.', color='#ff7f0e')

    magnetizations_mf = np.linspace(min(magnetizations), max(magnetizations), 1000)
    couplings_mf = lambda m: np.log((1+m)/(1-m))/(8*m)

    ax[0].plot(couplings_mf(magnetizations_mf), magnetizations_mf, label='Exact')
    ax[0].legend()
    plt.show()


def monte_carlo_2_4e():
    num_runs = 1000
    lattice_side = 10
    couplings = np.linspace(0, 1, 20)
    mc_ferromagnet = monte_carlo.FerroMagnet(num_runs, 0, 0, lattice_side)
    mc_ferromagnet.equilibration_time = 100
    fig, ax = plt.subplots(1)
    ax.set_xlabel(r"$J$")
    ax.grid("on")
    ax.set_ylabel(r"$m$")

    for coupling in couplings:
        magnetizations = []
        mc_ferromagnet.coupling = coupling
        for _ in range(5):
            mc_ferromagnet.reset()
            mc_ferromagnet.simulate_unit()
            magnetization = mc_ferromagnet.mean_magnetization()[0]
            magnetizations.append(magnetization)
        ax.plot(np.ones(len(magnetizations))*coupling, magnetizations, '.', color='#1f77b4')

    magnetizations_mf = np.linspace(-0.999, 0.999, 1000)
    couplings_mf = lambda m: np.log((1+m)/(1-m))/(8*m)

    ax.plot(couplings_mf(magnetizations_mf), magnetizations_mf, label='Mean field result')
    ax.legend()
    plt.show()


def monte_carlo_2_5():
    lattice_side, field = 10, 0
    couplings = [0.3, 0.4, 0.6]
    num_runs = [1000, 100000, 1000]
    mc_ferromagnet = monte_carlo.FerroMagnet(0, field, 0, lattice_side)
    mc_ferromagnet.equilibration_time = 100

    fig, ax = plt.subplots(1, 3)
    for idx in range(len(couplings)):
        mc_ferromagnet.coupling = couplings[idx]
        mc_ferromagnet.num_runs = num_runs[idx]
        mc_ferromagnet.simulate_unit()
        ax[idx].hist(mc_ferromagnet.magnetizations/lattice_side**2)

    plt.show()


def monte_carlo_3_1a():
    field, num_runs = 0, 1000
    lattice_sides = [5, 10, 15, 20]
    mc_ferromagnet = monte_carlo.FerroMagnet(num_runs, field, 0, 0)
    mc_ferromagnet.equilibration_time = 100
    couplings = [0.3, 0.5]
    for coupling in couplings:
        mc_ferromagnet.coupling = coupling
        for lattice_side in lattice_sides:
            mc_ferromagnet.lattice_side = lattice_side
            mc_ferromagnet.simulate_unit()
            fig = mc_ferromagnet.plot_results()[0]
            fig.suptitle("J = %f" % coupling)
    plt.show()


def monte_carlo_3_2a():
    lattice_side, num_runs = 10, 1000
    repeat = 3
    couplings = np.linspace(0, 1, 50)
    mc_ferromagnet = monte_carlo.FerroMagnet(num_runs, 0, 0, lattice_side)
    fig, ax = plt.subplots(1)
    mean_magnetization = []
    mean_magnetization_abs = []
    mc_ferromagnet.equilibration_time = 100
    for coupling in tqdm(couplings):
        for _ in range(repeat):
            mc_ferromagnet.coupling = coupling
            mc_ferromagnet.simulate_unit(pbar=False)
            mean_magnetization.append(mc_ferromagnet.mean_magnetization()[0]**2)
            mean_magnetization_abs.append(mc_ferromagnet.mean_magnetization(absolute=True)[0]**2)

    couplings = list(itertools.chain.from_iterable([[coupling]*repeat for coupling in couplings]))

    ax.plot(couplings, mean_magnetization, '.', label=r'$\langle m \rangle$^2', color='#1f77b4')
    ax.plot(couplings, mean_magnetization_abs, '.', label =r'$\langle |m| \rangle^2$', color='#ff7f0e')
    ax.set_xlabel(r"$J$")
    ax.set_ylabel(r"$m^2$")
    ax.legend()
    ax.grid('on')
    plt.show()


def monte_carlo_3_2d():

    def set_labels(axis_list, xlabel, ylabel):
        for n in range(2):
            for m in range(2):
                axis_list[n, m].set_xlabel(xlabel)
                axis_list[n, m].set_ylabel(ylabel)
                axis_list[n, m].grid('on')

    num_runs, field = 10000, 0
    lattice_sides = [5, 10, 15, 20]
    couplings = np.linspace(0.2, 0.6, 50)
    critical_point_index = abs(couplings - 0.4).argmin() + 1
    mc_ferromagnet = monte_carlo.FerroMagnet(num_runs, field, 0, 0)
    mc_ferromagnet.equilibration_time = 500

    # Figures:
    fig_m, ax_m = plt.subplots(2, 2)
    fig_power, ax_power = plt.subplots(2, 2)
    fig_susc, ax_susc = plt.subplots(2, 2)

    set_labels(ax_m, r"$J$", r"$\langle |m| \rangle$")
    set_labels(ax_susc, r"$J$", r"$\chi'/N$")
    set_labels(ax_power, r"$\ln(\langle |m| \rangle)$", r"$\ln(J - J_c)$")

    for idx, lattice_side in enumerate(lattice_sides):
        mc_ferromagnet.lattice_side = lattice_side
        magnetizations = []
        susceptibilities = []
        for coupling in tqdm(couplings):
            mc_ferromagnet.coupling = coupling
            mc_ferromagnet.simulate_unit(pbar=False)
            mean_magnetization = mc_ferromagnet.mean_magnetization(absolute=True)[0]
            magnetizations.append(mean_magnetization)
            susceptibility = mc_ferromagnet.susceptibility(absolute=True)
            susceptibilities.append(susceptibility)
        magnetizations = np.array(magnetizations)

        ax_m[0 if idx < 2 else 1, idx % 2].plot(couplings, magnetizations, '.')
        ax_m[0 if idx < 2 else 1, idx % 2].set_title("L = %i" % lattice_side)
        ax_power[0 if idx < 2 else 1, idx % 2]\
            .plot(np.log(magnetizations[critical_point_index:]), np.log(couplings[critical_point_index:] - 0.4), '.')
        ax_power[0 if idx < 2 else 1, idx % 2].set_title("L = %i" % lattice_side)
        ax_susc[0 if idx < 2 else 1, idx % 2].plot(couplings, susceptibilities, '.')
        ax_susc[0 if idx < 2 else 1, idx % 2].set_title("L = %i" % lattice_side)

    plt.show()


def monte_carlo_3_3a():
    lattice_side = 20
    num_runs = 10000
    couplings = [0.3, 0.4, 0.5]
    mc_ferromagnet = monte_carlo.FerroMagnet(num_runs, 0, 0, lattice_side)
    mc_ferromagnet.equilibration_time = 100

    fig, ax = plt.subplots(1)
    for coupling in couplings:
        mc_ferromagnet.coupling = coupling
        mc_ferromagnet.simulate_unit()
        mc_ferromagnet.plot_correlation(ax, label=r"$J = %2.1f$" % coupling)

    ax.grid('on')
    ax.legend()
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$g(r)$")
    plt.show()
