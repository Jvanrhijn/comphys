import numpy as np
import matplotlib.pyplot as plt
import lib.util as util
import unittest


class MonteCarlo(object):
    """Base Monte Carlo simulator, defines virtual interface"""
    def __init__(self, num_runs):
        self._num_runs = num_runs

    def init_state(self):
        pass

    def plot_state(self, *args, **kwargs):
        pass

    def variable_error(self):
        pass


class IsingModel(MonteCarlo):
    """Ising model solver using Monte Carlo method"""
    def __init__(self, num_runs, magnetic_field, lattice_side):
        super().__init__(num_runs)
        self._magnetic_field = magnetic_field
        self._lattice_side = lattice_side

