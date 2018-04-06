import math
import copy
import itertools
import lib.molecular_dynamics as md
from decorators.decorators import *
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda *i, **kwargs: i[0]
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 20})
rc('text', usetex=True)


def save_figure(fname) -> None:
    plt.savefig('/home/jesse/Dropbox/Uni/Jaar 3/Computational physics/molecular_dynamics/'+fname+'.pdf', format='pdf')


def force_coupled_sho(state) -> np.ndarray:
    return -2*state.positions + np.roll(state.positions, 1, axis=1) + np.roll(state.positions, -1, axis=1)


def force_lennard_jones_mic(state, cutoff, box_side) -> np.ndarray:
    """Lennard-Jones force between all particles in state, for a cubic box"""
    center = 0.5*box_side
    # Apply mimimal image criterion
    shifted = (state.positions[:, :, np.newaxis] - (state.positions
                                                    - center*np.ones(state.dim)[:, np.newaxis])[:, np.newaxis, :]).T \
        % box_side
    separation_mat = np.ones(shifted.shape)*center - shifted
    dist_mat = np.sqrt(np.einsum('ijk,ijk->ij', separation_mat, separation_mat, optimize='optimal'))
    # Kill self-interaction and apply cutoff radius
    np.fill_diagonal(dist_mat, np.inf)
    np.where(dist_mat > cutoff, np.inf, dist_mat)
    # Calculate f_ij and F_i
    force = np.einsum('ij,ijk->ki',
                      24*(2/dist_mat**14 - 1/dist_mat**8), separation_mat, optimize='optimal')
    # Compute state variables
    state.potential_energy = (2*(1/dist_mat**12 - 1/dist_mat**6) - 2*(1/cutoff**12 - 1/cutoff**6)).sum()
    state.pressure = 12*(2/dist_mat**12 - 1/dist_mat**6).sum()/(3*box_side**3)
    return force


def distribution_function(state, box_side, bins):
    center = 0.5*box_side
    # Apply mimimal image criterion
    shift_by = state.positions - np.array([center]*state.dim)[:, np.newaxis]
    shifted = (state.positions[:, :, np.newaxis] - shift_by[:, np.newaxis, :]).T % box_side
    separation_mat = np.ones(shifted.shape)*center - shifted
    dist_mat = np.linalg.norm(separation_mat, axis=2)
    return np.histogram(dist_mat, bins=bins, normed=False)[0]/state.num_particles


def molecular_dynamics_1_1a():
    dt = 2*((5*10**-3)/(8*np.pi))**(1/3)
    num_steps = int(math.ceil(8*np.pi/dt))
    init_state = md.State(1, dim=1)
    init_state.positions = np.array([[1.]])

    sim = md.Simulator(init_state, md.VerletIntegrator, dt, num_steps, lambda s: -s.positions)
    sim.save = True
    sim.set_state_vars(("Kinetic energy", lambda s: s.kinetic_energy()),
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


def molecular_dynamics_1_1b():
    dt = ((5*10**-3)/(4*np.pi))**(1/3)
    num_steps = int(math.ceil(4*np.pi/dt))
    time = np.linspace(dt, num_steps*dt, num_steps)

    position_range = [(-0.1, 0.1), (-1, 1), (-math.sqrt(0.5), math.sqrt(0.5))]
    velocity_range = [(-1, 1), (-0.1, 0.1), (-math.sqrt(0.5), math.sqrt(0.5))]
    fig, ax = plt.subplots(1, len(position_range), sharey=True, figsize=(20, 5))
    for idx, (pos_range, vel_range) in enumerate(zip(position_range, velocity_range)):
        sim = md.Simulator(md.State(100, dim=1).init_random(pos_range, vel_range),
                           md.VerletIntegrator, dt, num_steps, lambda s: -s.positions)

        sim.set_state_vars(("Kinetic energy", lambda s: s.kinetic_energy()),
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


def molecular_dynamics_1_1c():
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
        sim.set_state_vars(("Kinetic energy", lambda s: s.kinetic_energy()),
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
    sim.set_state_vars(("Temperature", lambda s: s.temperature()))
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
    fig, ax = plt.subplots(1, figsize=(20, 7))
    ax.hist(speeds, bins=50, normed=True)
    boltzmann = np.exp(-np.sort(speeds)**2/(2*temperature))
    boltzmann /= np.trapz(boltzmann, np.sort(speeds))
    ax.plot(np.sort(speeds), boltzmann,
            label=r"Boltzmann")  # must sort speeds to prevent crash, probably bug in matplotlib
    ax.set_xlabel(r"$v$"), ax.legend()
    save_figure("1c_iv")

    plt.show()


def molecular_dynamics_1_2d():

    num_particles = 125
    end_time = 10
    dt = (10**-6/end_time)**(1/3)
    time = np.arange(dt, end_time, dt)
    num_steps = len(time)

    density = 0.75
    box_side = (num_particles/density)**(1/3)

    box = md.Box(box_side, box_side, box_side)

    state = md.State(num_particles).init_random((0, box.side(0)), (0, 10))
    state.velocities -= state.center_of_mass()[1]
    state.set_temperature(3)
    state.init_grid(box)

    force = lambda s: force_lennard_jones_mic(s, 2.5, box.side(0))

    sim = md.BoxedSimulator(state, md.VerletIntegrator, dt, num_steps, force, box)
    sim.set_state_vars(("Temperature", lambda s: s.temperature()),
                       ("Kinetic", lambda s: s.kinetic_energy()),
                       ("Potential", lambda s: s.potential_energy))

    """
    vis = md.Visualizer(sim, inf_sim=True)
    fig, ax, ani = vis.particle_cloud_animation(100, 1,
                                                xaxis_bounds=(0, box_side),
                                                yaxis_bounds=(0, box_side),
                                                zaxis_bounds=(0, box_side))
    plt.show()
    """

    sim.save = True
    sim.simulate()

    kinetic, potential = sim.state_vars["Kinetic"], sim.state_vars["Potential"]
    energy = kinetic + potential
    energy_start = energy[0]
    temperature = sim.state_vars["Temperature"]

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].plot(time, (energy - energy_start)/energy_start)
    ax[1].plot(time, kinetic, label="Kinetic"), ax[1].plot(time, potential, label="Potential"), \
    ax[1].plot(time, energy, label="Total"), ax[1].legend()
    ax[2].plot(time, temperature)

    ax[0].set_ylabel(r"$(E(t) - E_0)/E_0"), ax[1].set_ylabel(r"Energy"), ax[2].set_ylabel(r"$T(t)$")
    for axis in ax:
        axis.set_xlabel(r"$t$")
        axis.grid()
    fig.tight_layout()
    save_figure("2d_ii")

    equilibration = len(time)//10
    temp = np.mean(temperature[equilibration:])
    print("Temperature T = {:.2f}".format(temp))

    speeds = np.array(list(itertools.chain.from_iterable(
        np.array([abs(state.velocities[0, :]) for state in sim.states][equilibration:]))))
    fig, ax = plt.subplots(1, figsize=(20, 7))
    ax.hist(speeds, bins=50, normed=True)
    boltzmann = np.exp(-np.sort(speeds)**2/(2*temp))  # Factor 3 in temp/3 needed to account for single-dimension
    boltzmann /= np.trapz(boltzmann, np.sort(speeds))
    ax.plot(np.sort(speeds), boltzmann,
            label=r"Boltzmann")  # must sort speeds to prevent crash, probably bug in matplotlib
    ax.set_xlabel(r"$|v_x|$"), ax.legend()
    save_figure("2d_iii")

    plt.show()


def molecular_dynamics_2_1():
    num_particles = 125
    density = 0.75
    temp_init = 2
    box_side = (num_particles/density)**(1/3)
    box = md.Box(box_side, box_side, box_side)
    cutoff = 2.5

    end_time = 10
    dt = (10**-6/end_time)**(1/3)
    time = np.arange(dt, end_time, dt)
    num_steps = len(time)

    state = md.State(num_particles).init_random((0, box_side), (0, 10))
    state.init_grid(box)
    state.velocities -= state.center_of_mass()[1]
    state.set_temperature(temp_init)

    for _ in range(5):
        sim = md.BoxedSimulator(state, md.VerletIntegrator, dt, num_steps,
                            lambda s: force_lennard_jones_mic(s, cutoff, box.side(0)),
                            box)
        sim.set_state_vars(("Temperature", lambda s: s.temperature()))
        sim.simulate()

        equilibration = len(time)//10
        temperature = np.mean(sim.state_vars["Temperature"][equilibration:])
        print("\nTemperature: T = {:.2f}\n".format(temperature, 2))


def molecular_dynamics_2_2():
    num_particles = 125
    density = 0.7
    box_side = (num_particles/density)**(1/3)
    box = md.Box(box_side, box_side, box_side)
    cutoff = 2.5

    end_time = 10
    dt = (10**-6/end_time)**(1/3)
    time = np.arange(dt, end_time, dt)
    num_steps = len(time)
    num_prep_steps = num_steps//4

    for temp_init in [2.0, 0.9]:
        state = md.State(num_particles).init_random((0, box_side), (0, 10))
        state.init_grid(box)
        state.velocities -= state.center_of_mass()[1]
        state.set_temperature(temp_init)
        sim = md.BoxedNVESimulator(state, md.VerletIntegrator, dt, num_steps,
                                   lambda s: force_lennard_jones_mic(s, cutoff, box.side(0)),
                                   box, temp_init, num_prep_steps)
        sim.set_state_vars(("Temperature", lambda s: s.temperature()))
        sim.simulate()
        equilibration = len(time)//10 + num_prep_steps
        temperature = np.mean(sim.state_vars["Temperature"][equilibration:])

        print("\nTemperature: T = {:.2f}\n".format(temperature))


def molecular_dynamics_2_3a():
    num_particles = 125
    density = 0.8
    box_side = (num_particles/density)**(1/3)
    box = md.Box(box_side, box_side, box_side)
    init_temp = 2
    cutoff = 2.5

    end_time = 1
    dt = (10**-8/end_time)**(1/3)
    time = np.arange(dt, end_time, dt)
    num_steps = len(time)
    num_prep_steps = num_steps//4

    state = md.State(num_particles).init_random((0, box_side), (0, 10))
    state.init_grid(box)
    state.velocities -= state.center_of_mass()[1]
    state.set_temperature(init_temp)

    sim = md.BoxedNVESimulator(state, md.VerletIntegrator, dt, num_steps,
                               lambda s: force_lennard_jones_mic(s, cutoff, box.side(0)),
                               box, init_temp, num_prep_steps)
    sim.set_state_vars(("Pressure", lambda s: s.pressure),
                       ("Temperature", lambda s: s.temperature()))
    sim.simulate()

    equilibration = len(time)//10 + num_prep_steps
    temperature = np.mean(sim.state_vars["Temperature"][equilibration:])

    pressure = np.mean(sim.state_vars["Pressure"][equilibration:]) + density*temperature

    print("\nPressure: P = {0:.2f}\nIdeal gas law: P = {1:.2f}\nDeviation: {2:.2f}".format(pressure, density*temperature,
          pressure - density*temperature))


def molecular_dynamics_2_3b():
    num_particles = 125
    densities = np.linspace(.01, 0.8, 40)
    init_temp = 2
    cutoff = 2.5

    end_time = 1
    dt = (10**-6/end_time)**(1/3)
    time = np.arange(dt, end_time, dt)
    num_steps = len(time)
    num_prep_steps = num_steps//4

    pressures, temperatures = [], []
    for density in tqdm(densities):
        box_side = (num_particles/density)**(1/3)
        box = md.Box(box_side, box_side, box_side)
        state = md.State(num_particles).init_random((0, box_side), (0, 10))
        state.init_grid(box)
        state.velocities -= state.center_of_mass()[1]
        state.set_temperature(init_temp)
        sim = md.BoxedNVESimulator(state, md.VerletIntegrator, dt, num_steps,
                                   lambda s: force_lennard_jones_mic(s, cutoff, box.side(0)),
                                   box, init_temp, num_prep_steps)
        sim.set_state_vars(("Pressure", lambda s: s.pressure),
                           ("Temperature", lambda s: s.temperature()))
        sim.simulate()
        equilibration = len(time)//10 + num_prep_steps
        temperature = np.mean(sim.state_vars["Temperature"][equilibration:])
        pressure = np.mean(sim.state_vars["Pressure"][equilibration:]) + density*temperature
        pressures.append(pressure)
        temperatures.append(temperature)
    pressure = np.array(pressures)
    temperatures = np.array(temperatures)

    fig, ax = plt.subplots(1, figsize=(20, 7))
    ax.plot(densities, pressure, '.', label="MD with virial formula")
    ax.plot(densities, np.array(densities)*temperatures, label="Ideal gas law")
    ax.grid()
    ax.set_ylabel(r"$P$"), ax.set_xlabel(r"$\rho$")
    ax.legend()
    fig.tight_layout()

    save_figure("2_3b")

    plt.show()


def molecular_dynamics_2_3c():
    num_particles = 125
    densities = np.linspace(0.4, 0.8, 40)
    init_temp = 0.8
    cutoff = 2.5

    end_time = 1
    dt = (10**-6/end_time)**(1/3)
    time = np.arange(dt, end_time, dt)
    num_steps = len(time)
    num_prep_steps = num_steps//4

    pressures, temperatures = [], []
    for density in tqdm(densities):
        box_side = (num_particles/density)**(1/3)
        box = md.Box(box_side, box_side, box_side)
        state = md.State(num_particles).init_random((0, box_side), (0, 10))
        state.init_grid(box)
        state.velocities -= state.center_of_mass()[1]
        state.set_temperature(init_temp)
        sim = md.BoxedNVESimulator(state, md.VerletIntegrator, dt, num_steps,
                                   lambda s: force_lennard_jones_mic(s, cutoff, box.side(0)),
                                   box, init_temp, num_prep_steps)
        sim.set_state_vars(("Pressure", lambda s: s.pressure),
                           ("Temperature", lambda s: s.temperature()))
        sim.simulate()
        equilibration = len(time)//10 + num_prep_steps
        temperature = np.mean(sim.state_vars["Temperature"][equilibration:])
        pressure = np.mean(sim.state_vars["Pressure"][equilibration:]) + density*temperature
        pressures.append(pressure)
    pressure = np.array(pressures)

    fig, ax = plt.subplots(1, figsize=(20, 7))
    ax.plot(densities, pressure, '.')
    ax.grid()
    ax.set_ylabel(r"$P$"), ax.set_xlabel(r"$\rho$")
    fig.tight_layout()

    save_figure("2_3c")

    plt.show()


def molecular_dynamics_2_4():
    num_particles = 125
    density = 0.8
    init_temp = 2
    cutoff = 2.5

    end_time = 1
    dt = (10**-8/end_time)**(1/3)
    time = np.arange(dt, end_time, dt)
    num_steps = len(time)
    num_prep_steps = num_steps//4

    box_side = (num_particles/density)**(1/3)
    box = md.Box(box_side, box_side, box_side)
    state = md.State(num_particles).init_random((0, box_side), (0, 10))
    state.init_grid(box)
    state.velocities -= state.center_of_mass()[1]
    state.set_temperature(init_temp)
    sim = md.BoxedNVESimulator(state, md.VerletIntegrator, dt, num_steps,
                               lambda s: force_lennard_jones_mic(s, cutoff, box.side(0)),
                               box, init_temp, num_prep_steps)
    sim.save = True
    sim.set_state_vars(("Potential", lambda s: s.potential_energy),
                       ("Temperature", lambda s: s.temperature()),
                       ("Pressure", lambda s: s.pressure))
    sim.simulate()

    # Calculate distribution function
    equilibration = len(time)//10 + num_prep_steps
    dr = box_side/125  # arbitrary choice of bin size
    bins = np.arange(dr, box_side/2, dr)
    distr = 0
    for idx, state in enumerate(sim.states[equilibration:]):
        distr += distribution_function(state, box_side, bins=bins)
    # divide distribution function by number of steps and number of particles
    distr = distr/(num_steps - equilibration)/density

    r = np.linspace(bins[0], bins[-1], len(distr))
    norm_vol = 4/3*np.pi*((r + dr)**3 - r**3)

    fig, ax = plt.subplots(1, figsize=(20, 7))
    ax.plot(r/box_side, distr/norm_vol, 'o')
    ax.set_xlabel(r"$r/L$"), ax.set_ylabel(r"$g(r)$")
    ax.grid()

    fig.tight_layout()
    save_figure("2_4a")

    # Compute average interaction energy:
    interaction_energy = np.trapz(density*distr*(4*(1/r**12 - 1/r**6))*4*np.pi*r**2, r)
    print("Average interaction energy: {:.2f}".format(interaction_energy))

    # Compute average potential energy
    potential_energy = np.mean(sim.state_vars["Potential"][equilibration:])
    print("Average potential energy - distribution function:\n{0:.2f}\n".format(interaction_energy*num_particles/2))
    print("Average potential energy - direct:\n{0:.2f}\n".format(potential_energy))

    # Compute pressure
    temperature = np.mean(sim.state_vars["Temperature"][equilibration:])
    pressure = density*temperature - 0.5*np.trapz(density*distr*24*(2*r**-13 - r**-7)
                                                                * r*r**2*4*np.pi, r)
    pressure_direct = np.mean(sim.state_vars["Pressure"][equilibration:]) + density*temperature
    print("Average pressure - distribution function:\n{0:.2f}\n".format(pressure))
    print("Average pressure - direct:\n{0:.2f}\n".format(pressure_direct))

    plt.show()


def molecular_dynamics_2_4f():
    num_particles = 125
    init_temp = 2
    cutoff = 2.5

    end_time = 1
    dt = (10**-8/end_time)**(1/3)
    time = np.arange(dt, end_time, dt)
    num_steps = len(time)
    num_prep_steps = num_steps//4

    equilibration = len(time)//10 + num_prep_steps
    fig, ax = plt.subplots(1, figsize=(20, 7))

    for density in [0.1, 0.8]:
        box_side = (num_particles/density)**(1/3)
        box = md.Box(box_side, box_side, box_side)
        state = md.State(num_particles).init_random((0, box_side), (0, 10))
        state.init_grid(box)
        state.velocities -= state.center_of_mass()[1]
        state.set_temperature(init_temp)
        sim = md.BoxedNVESimulator(state, md.VerletIntegrator, dt, num_steps,
                                   lambda s: force_lennard_jones_mic(s, cutoff, box.side(0)),
                                   box, init_temp, num_prep_steps)
        sim.save = True
        sim.simulate()

        # Calculate distribution function
        dr = box_side/125  # arbitrary choice of bin size
        bins = np.arange(dr, box_side/2, dr)
        distr = 0
        for idx, state in enumerate(sim.states[equilibration:]):
            distr += distribution_function(state, box_side, bins=bins)
        # divide distribution function by number of steps and number of particles
        distr = distr/(num_steps - equilibration)/density

        r = np.linspace(bins[0], bins[-1], len(distr))
        norm_vol = 4/3*np.pi*((r + dr)**3 - r**3)

        ax.plot(r/box_side, distr/norm_vol, 'o', label=r"$\rho = {}$".format(density))

    ax.legend()
    ax.set_xlabel(r"$r/L$"), ax.set_ylabel(r"$g(r)$")
    ax.grid()
    fig.tight_layout()

    save_figure("2_4f")
    plt.show()




