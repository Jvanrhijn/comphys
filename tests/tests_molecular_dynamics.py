import unittest
import cmath
import numpy as np
import matplotlib.pyplot as plt
import numpy.testing as nptest
import lib.molecular_dynamics as md


class TestState(unittest.TestCase):

    def test_init(self):
        test_state = md.State(10)  # Default init test
        defaults = np.zeros((3, 10))
        nptest.assert_array_equal(test_state.positions, defaults)
        nptest.assert_array_equal(test_state.velocities, defaults)
        pos_range = (-2, 2)
        vel_range = (-1, 1)
        self.assertTrue((abs(md.State(10).init_random(pos_range, vel_range).positions) < 2).all())
        self.assertTrue((abs(md.State(10).init_random(pos_range, vel_range).velocities) < 1).all())

    def test_single_particle(self):
        test_state = md.State(10).init_random((-1, 1), (-1, 1))
        zero = test_state.get_single_particle(0)
        one = test_state.get_single_particle(1)
        nptest.assert_array_equal(zero.positions, test_state.positions[:, 0])
        nptest.assert_array_equal(one.velocities, test_state.velocities[:, 1])

    def test_integrator_step(self):
        no_force = lambda state: 0
        constant_force = lambda state: 1
        sho_force = lambda state: -state.positions
        init_state = md.State(1, dim=1)
        dt = 1
        integrator = md.VerletIntegrator(init_state, no_force, dt)
        next_step = next(integrator)
        nptest.assert_array_equal(next_step.positions, md.State(1, dim=1).positions)  # For no force, state remains
        nptest.assert_array_equal(next_step.velocities, md.State(1, dim=1).velocities)

        integrator = md.VerletIntegrator(init_state, constant_force, dt)
        next_step = next(integrator)
        after_constant = md.State(1, dim=1)
        after_constant.positions = np.array([[0.5]])
        after_constant.velocities = np.array([[1.]])
        nptest.assert_array_almost_equal(next_step.positions, after_constant.positions)
        nptest.assert_array_almost_equal(next_step.velocities, after_constant.velocities)

        init_state = md.State(1, dim=1)
        init_state.positions = np.array([[1.]])
        integrator = md.VerletIntegrator(init_state, sho_force, dt)
        next_step = next(integrator)
        nptest.assert_array_almost_equal(next_step.positions, np.array([[0.5]]))
        nptest.assert_array_almost_equal(next_step.velocities, np.array([[-0.75]]))

    def test_integrator_full_simulation(self):
        force = lambda state: -state.positions  # SHO
        position_analytical = lambda t: cmath.cos(t)

        num_steps = 100
        dt = (1/(10**8*num_steps))**0.25  # Sufficiently small time step for error at most 10**-7
        end_time = num_steps*dt
        state = md.State(1, dim=1)
        state.positions = np.array([[1.]])
        integrator = md.VerletIntegrator(state, force, dt)

        for n in range(0, num_steps):
            next(integrator)
        self.assertAlmostEqual(state.positions[0, 0], position_analytical(end_time))


class TestSimulator(unittest.TestCase):

    def test_init(self):
        sim = md.MDSimulator(md.State(1, dim=1), md.VerletIntegrator, 0.001, 1000, lambda pos: 0)
        self.assertEqual(md.State(1, dim=1).positions, sim._integrator._state.positions)
        self.assertEqual(md.State(1, dim=1).velocities, sim._integrator._state.velocities)
        self.assertAlmostEqual(0.001*1000, sim._end_time)

    def test_simulation(self):
        position_analytical = lambda t: cmath.cos(t)
        num_steps = 100
        dt = (1/(10**8*num_steps))**0.25  # Sufficiently small time step for error at most 10**-7
        init_state = md.State(1, dim=1)
        init_state.positions = np.array([[1.]])
        sim = md.MDSimulator(init_state, md.VerletIntegrator, dt, num_steps, lambda state: -state.positions)
        self.assertAlmostEqual(sim.simulate().positions[0, 0], position_analytical(sim._end_time))

    def test_state_vars(self):
        energy = lambda s: 0.5*np.sum(s.positions**2 + s.velocities**2)
        num_steps = 100
        dt = (1/(10**4*num_steps))**0.25  # Sufficiently small time step for error at most 10**-7
        init_state = md.State(1, dim=1)
        init_state.positions = np.array([[1.]])
        sim = md.MDSimulator(init_state, md.VerletIntegrator, dt, num_steps, lambda state: -state.positions)
        sim.save = True
        sim.set_state_vars(("energy", energy))
        sim.simulate()
        nptest.assert_array_almost_equal(sim.state_vars["energy"], np.array([energy(state) for state in sim.states]))

    def test_energy_conservation(self):
        energy = lambda s: 0.5*np.sum(s.positions**2 + s.velocities**2)
        num_steps = 100
        dt = (1/(10**8*num_steps))**0.25  # Sufficiently small time step for error at most 10**-8
        init_state = md.State(1, dim=1)
        init_state.positions = np.array([[1.]])
        sim = md.MDSimulator(init_state, md.VerletIntegrator, dt, num_steps, lambda state: -state.positions)
        sim.set_state_vars(("energy", energy))
        sim.simulate()
        exact_energy = 0.5  # Energy is conserved for a harmonic oscillator
        self.assertLess(max(abs(exact_energy - sim.state_vars["energy"]))/exact_energy, 10**-6)

