import math
import numpy as np
import matplotlib.pyplot as plt
import lib.molecular_dynamics as md
from decorators.decorators import *
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 16})
rc('text', usetex=True)


@plot_grid_show
def molecular_dynamics1a():
    dt = 0.4*(1/10**3*8*np.pi)**(1/3)
    num_steps = int(math.ceil(8*np.pi/dt))
    init_state = md.State(1, dim=1)
    init_state.positions = np.array([[1.]])

    sim = md.MDSimulator(init_state, md.VerletIntegrator, dt, num_steps, lambda s: -s.positions)
    sim.save = True
    sim.set_state_vars(("Energy", lambda s: 0.5*(np.sum(s.positions**2 + s.velocities**2))))
    sim.simulate()

    time = np.linspace(dt, num_steps*dt, num_steps)
    fig, ax = plt.subplots(2, sharex=True)
    positions = np.array([state.positions[0] for state in sim.states])
    ax[0].plot(time, np.cos(time), '.', label="Analytical"), ax[0].set_ylabel(r"$x(t)$")
    ax[0].plot(time, positions, label="Verlet")
    ax[1].plot(time, (abs(sim.state_vars["Energy"] - 0.5)/0.5)*1000), ax[1].set_ylabel(r"(E(t) - E_0)/E_0")
    ax[1].set_xlabel(r"$t$"),
    ax[0].legend()
    return ax
