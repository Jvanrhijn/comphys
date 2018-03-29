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
    return -(2*state.positions - np.roll(state.positions, 1, axis=1) - np.roll(state.positions, -1, axis=1))


def molecular_dynamics1a():
    dt = 0.4*(1/10**3*8*np.pi)**(1/3)
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

    ax[0, 1].plot(time, (abs(energy - 0.5)/0.5)*1000), ax[0, 1].set_ylabel(r"$(E(t) - E_0)/E_0$, $10^{-3}$")
    ax[1, 1].plot(time, sim.state_vars["Kinetic energy"], label="Kinetic energy")
    ax[1, 1].plot(time, sim.state_vars["Potential energy"], label="Potential energy")
    ax[1, 1].set_ylabel("Energy")
    ax[1, 1].set_xlabel(r"$t$"),

    ax[0, 0].legend(), ax[1, 0].legend(), ax[1, 1].legend()
    for axis in itertools.chain.from_iterable(ax):
        axis.grid()
    save_figure("1a")
    plt.show()


def molecular_dynamics1c():
    dt = 0.4*(1/10**3*8*np.pi)**(1/3)
    num_steps = int(math.ceil(8*np.pi/dt))
    init_state = md.State(10, dim=2).init_random((-1, 1), (-1, 1))

    sim = md.Simulator(init_state, md.VerletIntegrator, dt, num_steps, force_coupled_sho)
    energy = lambda s: 0.5*(np.sum(s.velocities**2 + (np.roll(s.positions, 1, axis=1) - s.positions)**2))
    sim.set_state_vars(("Energy", energy))
    sim.simulate()

    time = np.linspace(dt, num_steps*dt, num_steps)
    fig, ax = plt.subplots(1, figsize=(10, 5))
    ax.plot(time, sim.state_vars["Energy"])
    plt.show()
