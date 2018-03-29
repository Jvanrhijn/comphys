import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import lib.molecular_dynamics as md
from decorators.decorators import *
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 20})
rc('text', usetex=True)


def save_figure(fname):
    plt.savefig('/home/jesse/Dropbox/Uni/Jaar 3/Computational physics/molecular_dynamics/'+fname+'.pdf', format='pdf')


def force_coupled_sho(state):
    return -2*state.positions + np.roll(state.positions, 1, axis=1) + np.roll(state.positions, -1, axis=1)


def molecular_dynamics1a():
    dt = 2*((5*10**-3)/(8*np.pi))**(1/3)
    num_steps = int(math.ceil(8*np.pi/dt))
    init_state = md.State(1, dim=1)
    init_state.positions = np.array([[1.]])

    sim = md.Simulator(init_state, md.VerletIntegrator, dt, num_steps, lambda s: -s.positions)
    sim.save = True
    sim.set_state_vars(("Kinetic energy", lambda s: 0.5*(np.sum(s.velocities**2))),
                       ("Potential energy", lambda s: 0.5*np.sum(s.positions**2)))
    sim.simulate()

    energy = sim.state_vars["Kinetic energy"] + sim.state_vars["Potential energy"]

    time = np.linspace(dt, num_steps*dt, num_steps)
    fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharex='col')
    positions = np.array([state.positions[0] for state in sim.states])
    velocities = np.array([state.velocities[0] for state in sim.states])

    ax[0, 0].plot(time, np.cos(time), '.', label="Analytical"), ax[0, 0].set_ylabel(r"$x(t)$")
    ax[0, 0].plot(time, positions, label="Verlet")
    ax[1, 0].plot(time, -np.sin(time), '.', label="Analytical"), ax[1, 0].set_ylabel(r"$\dot{x}(t)$")
    ax[1, 0].plot(time, velocities, label="Verlet")
    ax[1, 0].set_xlabel(r"$t$"),

    ax[0, 1].plot(time, (energy - 0.5)/0.5*1000), ax[0, 1].set_ylabel(r"$(E(t) - E_0)/E_0$, $10^{-3}$")
    ax[1, 1].plot(time, sim.state_vars["Kinetic energy"], label="Kinetic energy")
    ax[1, 1].plot(time, sim.state_vars["Potential energy"], label="Potential energy")
    ax[1, 1].set_ylabel("Energy")
    ax[1, 1].set_xlabel(r"$t$"),

    ax[0, 0].legend(), ax[1, 0].legend(), ax[1, 1].legend()
    for axis in itertools.chain.from_iterable(ax):
        axis.grid()
    fig.tight_layout()
    save_figure("1a")
    plt.show()


def molecular_dynamics1b():
    dt = ((5*10**-3)/(4*np.pi))**(1/3)
    num_steps = int(math.ceil(4*np.pi/dt))
    time = np.linspace(dt, num_steps*dt, num_steps)

    position_range = [(-0.1, 0.1), (-1, 1), (-math.sqrt(0.5), math.sqrt(0.5))]
    velocity_range = [(-1, 1), (-0.1, 0.1), (-math.sqrt(0.5), math.sqrt(0.5))]
    fig, ax = plt.subplots(1, len(position_range), sharey=True, figsize=(20, 5))
    for idx, (pos_range, vel_range) in enumerate(zip(position_range, velocity_range)):
        sim = md.Simulator(md.State(100, dim=1).init_random(pos_range, vel_range),
                           md.VerletIntegrator, dt, num_steps, lambda s: -s.positions)

        sim.set_state_vars(("Kinetic energy", lambda s: 0.5*(np.sum(s.velocities**2))),
                           ("Potential energy", lambda s: 0.5*np.sum(s.positions**2)))
        sim.simulate()
        ax[idx].set_title(r"$x \in {0}$, $v \in {1}".format(
            tuple(map(lambda x: round(x, 3), list(pos_range))),
            tuple(map(lambda x: round(x, 3), list(vel_range)))))
        ax[idx].plot(time, sim.state_vars["Kinetic energy"], label="Kinetic energy")
        ax[idx].plot(time, sim.state_vars["Potential energy"], label="Potential energy")
    for axis in ax:
        axis.grid()
        axis.legend()
    fig.tight_layout()
    save_figure("1b_iii")
    plt.show()


def molecular_dynamics1c():
    dt = ((10**-4)/(8*np.pi))**(1/3)
    num_steps = int(math.ceil(8*np.pi/dt))
    time = np.linspace(dt, num_steps*dt, num_steps)
    num_particles = 100

    position_range = [(-0.1, 0.1), (-1, 1), (-math.sqrt(0.5), math.sqrt(0.5))]
    velocity_range = [(-1, 1), (-0.1, 0.1), (-math.sqrt(0.5), math.sqrt(0.5))]
    fig, ax = plt.subplots(1, len(position_range), sharey=True, figsize=(20, 5))
    for idx, (pos_range, vel_range) in enumerate(zip(position_range, velocity_range)):
        init_state = md.State(num_particles, dim=1).init_random(pos_range, vel_range)
        sim = md.Simulator(init_state, md.VerletIntegrator, dt, num_steps, force_coupled_sho)
        sim.state().velocities -= sim.state().center_of_mass()[1]  # Subtract translation mode
        sim.set_state_vars(("Kinetic energy", lambda s: 0.5*np.sum(s.velocities**2)),
                           ("Potential energy", lambda s: 0.5*np.sum((np.roll(s.positions, 1, axis=1)
                                                                      - s.positions)**2)))
        sim.simulate()
        kinetic = sim.state_vars["Kinetic energy"]
        potential = sim.state_vars["Potential energy"]
        energy = kinetic + potential
        ax[idx].set_title(r"$x \in {0}$, $v \in {1}".format(
            tuple(map(lambda x: round(x, 3), list(pos_range))),
            tuple(map(lambda x: round(x, 3), list(vel_range)))))
        ax[idx].plot(time, kinetic, label="Kinetic energy")
        ax[idx].plot(time, potential, label="Potential energy")
        ax[idx].plot(time, energy)
        ax[idx].set_xlabel(r"$t$")

    for axis in ax:
        axis.grid()
        axis.legend()
    fig.tight_layout()
    save_figure("1c_ii")

    sim = md.Simulator(md.State(num_particles, dim=1).init_random((-1, 1), (-1, 1)), md.VerletIntegrator,
                       dt, num_steps, force_coupled_sho)
    sim.state().velocities -= sim.state().center_of_mass()[1]
    sim.set_state_vars(("Temperature", lambda s: np.sum(s.velocities**2)/num_particles))
    sim.save = True
    sim.simulate()

    equilibration = len(time)//5
    temperature = np.mean(sim.state_vars["Temperature"][equilibration:])
    print("Temperature: T = {}".format(round(temperature, 2)))

    fig, ax = plt.subplots(1, figsize=(20, 5))
    ax.plot(time, sim.state_vars["Temperature"])
    ax.set_ylim(0), ax.set_xlabel(r"$t$"), ax.set_ylabel(r"$T(t)$")
    ax.grid()

    fig.tight_layout()
    save_figure("1c_iii")

    speeds = np.array(list(itertools.chain.from_iterable(
        np.array([np.linalg.norm(state.velocities, axis=0) for state in sim.states][equilibration:]))))
    fig, ax = plt.subplots(1, figsize=(20, 5))
    ax.hist(speeds, bins=50, normed=True)
    boltzmann = np.exp(-np.sort(speeds)**2/(2*temperature))
    boltzmann /= np.trapz(boltzmann, np.sort(speeds))
    ax.plot(np.sort(speeds), boltzmann,
            label=r"Boltzmann")  # must sort speeds to prevent crash, probably bug in matplotlib
    ax.set_xlabel(r"$v$"), ax.legend()
    save_figure("1c_iv")

    plt.show()
