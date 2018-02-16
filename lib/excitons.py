import numpy as np
import util.util as util
import matplotlib.pyplot as plt
import unittest


class WaveFunction(object):
    """This object represents a quantum mechanical wave function on a discrete grid"""
    def __init__(self, grid, values=None):
        self.grid = grid
        if values is None:
            self.values = np.zeros(len(grid))
        else:
            assert(len(grid) == len(values))
            self.values = values

    def __add__(self, other):
        """Add two wave functions together element-wise"""
        if np.array_equal(self.grid, other.grid):
            return WaveFunction(self.grid, values=(self.values + other.values))
        else:
            raise ValueError("Wave functions must be defined on the same grid")

    def __sub__(self, other):
        if np.array_equal(self.grid, other.grid):
            return WaveFunction(self.grid, values=(self.values - other.values))
        else:
            raise ValueError("Wave functions must be defined on the same grid")

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def __delitem__(self, index):
        self.values = np.delete(self.values, index)
        self.grid = np.delete(self.grid, index)

    def value(self, grid_point) -> float:
        """Returns value of wave function closest to the given grid point"""
        index = abs(self.grid - grid_point).argmin()
        return self.values[index]

    def normalize(self):
        """Normalizes the wave function"""
        norm = self.norm()
        self.values /= norm

    def norm(self) -> float:
        """Returns the norm of the wave function"""
        return np.sqrt(np.trapz(np.absolute(self.values)**2, self.grid))

    def plot(self, axis, *args, **kwargs):
        axis.plot(self.grid, self.values, *args, **kwargs)


class SchrodingerSolver(object):

    def __init__(self, grid, potential, eigenvalue, angular_momentum, boundary_left, boundary_right):
        """This class provides a solver for the Radial Schrödinger equation. The only public method is solve(), which
        does the forward and backward iterations, glues the two together, and returns a normalized solution for the
        given potential energy function.

        :param grid: grid to solve radial Schrödinger on
        :param potential: potential function to solve the equation for
        :param eigenvalue: eigenvalue to solve the equation for
        :param angular_momentum: angular momentum quantum number, for 'centrifugal' potential
        :param boundary_left: tuple of boundary values on the left side
        :param boundary_right: tuple of boundary values on the right side
        """
        self.grid = grid
        self.step_size = np.diff(grid)
        self.step_size = np.append(self.step_size, self.step_size[-1])
        self.potential = potential(grid)
        self.eigenvalue = eigenvalue
        self.angular_momentum = angular_momentum
        self.turning_point_index = abs(self.potential - eigenvalue).argmin()
        self.turning_point = self.grid[self.turning_point_index]
        self.boundary_left = boundary_left
        self.boundary_right = boundary_right

    def solve(self, propagator):
        """Solve the radial Schrödinger equation for the given potential, grid, and boundary conditions"""
        forward_generator = self.solve_forward(propagator)
        backward_generator = self.solve_backward(propagator)

        # Generate the forward and backward solutions
        solution_forward = np.array([value for value in forward_generator])
        solution_backward = np.flip(np.array([value for value in backward_generator]), 0)

        # Scale the two solutions at the turning point, then calculate the derivative continuity
        solution_forward /= solution_forward[-2]
        solution_backward /= solution_backward[1]
        derivative_continuity = solution_backward[2] - solution_backward[0] \
            - (solution_forward[-1] - solution_forward[-3])

        # Tie the two solutions together, overwriting the forward solution value at the turning point
        solution = np.concatenate((solution_forward[:-2],
                                  solution_backward[1:]))

        solution = WaveFunction(self.grid, values=solution)
        solution.normalize()

        return solution, derivative_continuity

    def solve_forward(self, propagator):
        p_previous = self.boundary_left[0]
        previous = self.boundary_left[1]
        index = 1
        while index < self.turning_point_index + 3:
            yield p_previous
            next_value = propagator(previous, p_previous, self.grid, self.potential, self.step_size,  self.eigenvalue,
                                    self.angular_momentum, index)
            index += 1
            p_previous, previous = previous, next_value

    def solve_backward(self, propagator):
        p_previous = self.boundary_right[1]
        previous = self.boundary_left[0]
        index = len(self.grid) - 2
        while index > self.turning_point_index - 3:
            # Pass in a reversed int, since the base propagation algorithms assume that the next value is index + 1.
            # In this case it is index - 1, so we can pass in the ReversedInt class, which reverses adding and
            # subtracting. A nice way to abuse Python's everything-is-an-object model.
            yield p_previous
            next_value = propagator(previous, p_previous, self.grid, self.potential, self.step_size,  self.eigenvalue,
                                    self.angular_momentum, ReversedInt(index))
            index -= 1
            p_previous, previous = previous, next_value

    @staticmethod
    def propagate_simple(previous, p_previous, grid, potential, step_size, eigenvalue, angular_momentum, index):
        """Return next value of solution, give two previous values and index, using the simple propagation algorithm"""
        next_value = 2*previous - p_previous \
            + step_size[index]**2*(potential[index] - eigenvalue)*previous
        return next_value

    @staticmethod
    def propagate_simple_log(previous, p_previous, grid, potential, step_size, eigenvalue, angular_momentum, index):
        """Return next value of solution, given two previous values and index,
        assuming a logarithmic grid
        """
        step_size = step_size[index]/grid[index]

        return (2*previous - p_previous + step_size**2*((0.5 + angular_momentum)**2 +
                                                        ((potential[index] - eigenvalue)
                                                         * grid[index]**2))*previous)

    @staticmethod
    def propagate_numerov(previous, p_previous, grid, potential, step_size, eigenvalue, angular_momentum, index):
        """Return next value of solution, given two previous values and index, using Numverov's method"""
        next_value = (5*step_size[index]**2/6*(
                potential[index] + angular_momentum*(angular_momentum + 1)/grid[index]**2
                - eigenvalue) + 2)*previous \
            - (1 - step_size[index-1]**2/12*(potential[index-1]
                                             + angular_momentum*(angular_momentum + 1)
                                             - eigenvalue))*p_previous
        next_value /= 1 - step_size[index+1]**2/12*(potential[index+1] - eigenvalue)
        return next_value

    @staticmethod
    def propagate_numerov_log(previous, p_previous, grid, potential, step_size, eigenvalue, angular_momentum, index):
        """Return next value of solution given two previous values ans index, using Numerov and
        assuming a logarithmic grid
        """
        # Transform step size for the propagation algorithm
        step_size = step_size[index]/grid[index]

        # Kind of ugly and inefficient to define this here, but at least it keeps the notation tidy
        def q(idx):
            return 1 - step_size**2/12*((angular_momentum + 0.5)**2
                                        + (potential[idx] - eigenvalue)*grid[idx]**2)

        next_value = (12 - 10*q(index))*previous - q(index-1)*p_previous
        next_value /= q(index+1)
        return next_value


class Shooter(object):
    """This class implements the shooting method of finding the correct eigenvalue"""
    def __init__(self, grid, potential, boundary_left, boundary_right, angular_momentum):
        self.grid = grid
        self.potential = potential
        self.boundary_left = boundary_left
        self.boundary_right = boundary_right
        self.angular_momentum = angular_momentum

    def get_solver(self, eigenvalue):
        """Returns a SchrodingerSolver initialized with the shooter parameters"""
        return SchrodingerSolver(self.grid, self.potential, eigenvalue, self.angular_momentum,
                                 self.boundary_left, self.boundary_right)

    def bisection_iteration(self, propagator, tolerance, left_bound, right_bound):
        """Do one iteration of the bisection algorithm, and check for convergence"""
        mid_point = 0.5*(left_bound + right_bound)
        solver_left = self.get_solver(left_bound)
        solver_mid = self.get_solver(mid_point)

        solution_left, derivative_continuity_left = solver_left.solve(propagator)
        solution_right, derivative_continuity_mid = solver_mid.solve(propagator)

        if np.sign(derivative_continuity_left) == np.sign(derivative_continuity_mid):
            new_mid_point = 0.5 * (mid_point + right_bound)
            return new_mid_point, abs(new_mid_point - mid_point) < tolerance, mid_point, right_bound
        else:
            new_mid_point = 0.5 * (left_bound + mid_point)
            return new_mid_point, abs(new_mid_point - mid_point) < tolerance, derivative_continuity_mid, \
                left_bound, mid_point

    def improved_iteration(self, propagator, tolerance, eigenvalue_guess):
        """Do one iteration of the improved algorithm"""
        solver = self.get_solver(eigenvalue_guess)
        solution, derivative_continuity = solver.solve(propagator)
        factor = solution[solver.turning_point_index]**2
        eigenvalue_guess -= factor*derivative_continuity/(2*solver.step_size[solver.turning_point_index])

        return eigenvalue_guess, abs(factor*derivative_continuity) < tolerance, derivative_continuity, \
            eigenvalue_guess

    @staticmethod
    def shooter(tolerance, propagator, iterator, *algorithm_inputs):
        """Do the shooting method of solving the Schrödinger equation"""
        converged = False
        derivative_continuities = []
        eigenvalues = []
        while not converged:
            eigenvalue, converged, derivative_continuity, \
             *algorithm_inputs = iterator(propagator, tolerance, *algorithm_inputs)
            derivative_continuities.append(derivative_continuity)
            eigenvalues.append(eigenvalue)
        return eigenvalues, derivative_continuities


class ReversedInt(int):
    """This class is an integer type with adding and subtracting reversed."""
    def __add__(self, other):
        return int.__sub__(self, other)

    def __sub__(self, other):
        return int.__add__(self, other)


class WaveFunctionTest(unittest.TestCase):
    """Tests for classses in this module"""
    def __init__(self, *args, **kwargs):
        super(WaveFunctionTest, self).__init__(*args, **kwargs)
        self.grid = np.linspace(0, 1, 100)
        self.values_first = np.random.rand(len(self.grid))
        self.values_second = np.random.rand(len(self.grid))
        self.wf_first = WaveFunction(self.grid, values=self.values_first)
        self.wf_second = WaveFunction(self.grid, values=self.values_second)

    def test_wf_value(self):
        self.assertEqual(self.values_first[40], self.wf_first.value(self.grid[40]))

    def test_wf_add(self):
        np.testing.assert_array_almost_equal(self.values_first + self.values_second,
                                             (self.wf_first + self.wf_second).values)
        with self.assertRaises(ValueError):
            wf_temp = WaveFunction(np.linspace(0, 2, 100), values=np.random.random(len(self.grid)))
            self.wf_first + wf_temp

    def test_wf_subtract(self):
        np.testing.assert_array_almost_equal(self.values_first - self.values_second,
                                             (self.wf_first - self.wf_second).values)
        with self.assertRaises(ValueError):
            wf_temp = WaveFunction(np.linspace(0, 2, 100), values=np.random.random(len(self.grid)))
            self.wf_first - wf_temp

    def test_wf_accessors(self):
        self.assertEqual(self.wf_first.values[0], self.wf_first[0])
        self.wf_first[10] = 1.0
        self.assertEqual(1.0, self.wf_first[10])
        del self.wf_first[-1]
        self.assertEqual(len(self.values_first) - 1, len(self.wf_first.values))
        self.assertEqual(self.values_first[-2], self.wf_first[-1])
        self.assertEqual(len(self.grid) - 1, len(self.wf_first.grid))

    def test_wf_normalize(self):
        self.wf_first.normalize()
        self.assertAlmostEqual(1.0, self.wf_first.norm(), 6)

    def test_wf_plot(self):
        fig, ax = plt.subplots()
        self.wf_first.plot(ax)
        self.wf_second.plot(ax)


class SolverTest(unittest.TestCase):
    def test_harmonic_oscillator(self):

        def potential(x):
            return 0.25*x**2
        grid = np.linspace(np.log(10**-5), np.log(20), 80000)
        grid = np.exp(grid)
        boundary_left = (grid[0], grid[1])
        boundary_right = (grid[-2]*np.exp(-grid[-2]**2/4), grid[-1]*np.exp(-grid[-1]**2/4))
        angular_momentum = 0
        solver = SchrodingerSolver(grid, potential, 1.5, angular_momentum, boundary_left, boundary_right)
        wave_function, derivative_continuity = solver.solve(solver.propagate_numerov_log)
        fig, ax = plt.subplots()
        wave_function.plot(ax)


class ShooterTest(unittest.TestCase):
    def test_shooter(self):

        def potential(x):
            return -2/x

        grid = np.linspace(np.log(10**-5), np.log(20), 8000)
        grid = np.exp(grid)
        boundary_left = (grid[0], grid[1])
        boundary_right = (grid[-2]*np.exp(-grid[-2]**2/4), grid[-1]*np.exp(-grid[-1]**2/4))
        angular_momentum = 0
        shooter = Shooter(grid, potential, boundary_left, boundary_right, angular_momentum)

        eigenvalues, derivative_continuities = shooter.shooter(10**-8, SchrodingerSolver.propagate_numerov_log,
                                                               shooter.improved_iteration, -0.45)
        table = util.generate_latex_table(eigenvalues, derivative_continuities, "$\lambda$", "$F(\lambda)$")


if __name__ == "__main__":
    unittest.main()
