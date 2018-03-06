import unittest
from lib.monte_carlo import *


class SpinConfigTest(unittest.TestCase):
    """Tests for SpinConfiguration class"""
    def test_initialize(self):
        some_lattice = np.array([[1, -1], [-1, -1]])
        config = SpinConfiguration(some_lattice)
        config_up = SpinConfiguration.all_up(5, 5)
        config_down = SpinConfiguration.all_down(5, 5)
        np.testing.assert_array_equal(config._lattice, some_lattice)
        np.testing.assert_array_equal(config_up._lattice, np.ones((5, 5)))
        np.testing.assert_array_equal(config_down._lattice, np.ones((5, 5))*-1)
        SpinConfiguration(SpinConfiguration.init_random(10, 10)._lattice)  # Check if init_random returns a valid lattice
        with self.assertRaises(ValueError):
            SpinConfiguration(np.array([[1, 2], [-1, 1]]))

    def test_accessors(self):
        some_lattice = np.array([[1, -1], [-1, -1]])
        config = SpinConfiguration(some_lattice)
        self.assertEqual(config[0, 0], 1)
        self.assertEqual(config[0, 1], -1)

    def test_magnetization(self):
        some_lattice = np.array([[1, -1], [-1, -1]])
        config = SpinConfiguration(some_lattice)
        magnetization = -2
        self.assertEqual(config.magnetization(), magnetization)

    def test_energy(self):
        configuration = SpinConfiguration(np.array([[1, -1], [-1, -1]]))
        magnetic_field, coupling = 1, 1
        energy = 2  # Manual calculation
        self.assertEqual(energy, configuration.energy(magnetic_field, coupling))

    def test_flip(self):
        some_lattice = np.array([[1, -1], [-1, -1]])
        flipped = np.array([[1, 1], [-1, -1]])
        configuration = SpinConfiguration(some_lattice)
        configuration.flip_spin(0, 1)
        np.testing.assert_array_equal(configuration._lattice, flipped)
        random_flipped = SpinConfiguration(some_lattice)
        row, column = random_flipped.flip_random()
        self.assertEqual(random_flipped[row, column], some_lattice[row, column]*-1)


class ParaMagnetTest(unittest.TestCase):
    """Tests for the paramagnet monte carlo solver"""
    def test_paramagnet(self):
        field = 1
        mc_paramagnet = ParaMagnet(2000, field, 10)
        mc_paramagnet.set_equilibration_time(200)
        exact_magnetization = np.tanh(field)
        mc_paramagnet.simulate()
        mean_magnetization, error = mc_paramagnet.mean_magnetization()
        stdev = error*np.sqrt(1999)
        # Test may fail in 5% of cases
        self.assertTrue(exact_magnetization - 2*stdev < mean_magnetization < exact_magnetization + 2*stdev)
        with self.assertRaises(AssertionError):
            mc_paramagnet.simulate()

    def test_reset(self):
        mc_paramagnet = ParaMagnet(2000, 1, 10)
        mc_paramagnet.set_equilibration_time(200)
        mc_paramagnet.simulate()
        magnetization_first = mc_paramagnet.mean_magnetization()[0]
        mc_paramagnet.reset()
        self.assertFalse(mc_paramagnet.is_done())
        # If reset worked, energies & magnetizations should both be zero and equal
        np.testing.assert_array_equal(mc_paramagnet.magnetizations[1:], mc_paramagnet.energies[1:])
        # After second simulation, new magnetization should be close to but not equal to previous simulation
        mc_paramagnet.simulate()
        mc_paramagnet.set_equilibration_time(200)
        magnetization_second = mc_paramagnet.mean_magnetization()[0]
        self.assertAlmostEqual(magnetization_first, magnetization_second, places=1)
        self.assertNotEqual(magnetization_first, magnetization_second)

    def test_set_functions(self):
        field = 1
        side = 10
        mc_paramagnet = ParaMagnet(2000, field, side)
        mc_paramagnet.set_magnetic_field(2)
        mc_paramagnet.set_lattice_side(11)
        mc_paramagnet.set_equilibration_time(10)
        self.assertEqual(11, mc_paramagnet._lattice_side)
        self.assertEqual(2, mc_paramagnet._magnetic_field)
        self.assertEqual(10, mc_paramagnet._equilibration_time)

    def test_unit_steps(self):
        field = 1
        side = 10
        num_units = 20
        mc_paramagnet = ParaMagnet(num_units, field, side)
        mc_paramagnet.simulate_unit()
        exact_magnetization = np.tanh(field)
        mc_paramagnet.set_equilibration_time(2)
        mean_magnetization, stdev = mc_paramagnet.mean_magnetization()
        self.assertEqual(len(mc_paramagnet.magnetizations), num_units+1)
        self.assertTrue(exact_magnetization - 2*stdev < mean_magnetization < exact_magnetization + 2*stdev)


class TestFerroMagnet(unittest.TestCase):
    """Tests for FerroMagnet class"""
    def test_ferromagnet(self):
        num_runs, field, coupling, lattice_side = 1000, 0, 0.35, 10
        mc_ferromagnet = FerroMagnet(num_runs, field, coupling, lattice_side)
        mc_ferromagnet.set_equilibration_time(num_runs//10)
        mc_ferromagnet.simulate_unit()
        mc_ferromagnet.plot_results()
        plt.show()


if __name__ == '__main__':
    unittest.main()
