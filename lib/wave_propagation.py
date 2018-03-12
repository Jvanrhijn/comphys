"""This module contains classes needed for the Wave Propagation project"""
import numpy as np


class BaseMatrix:

    def __init__(self, num_factors):
        self._num_factors = num_factors


class TransferMatrix(BaseMatrix):

    def __init__(self, num_factors):
        super().__init__(num_factors)


class ScatteringMatrix(BaseMatrix):

    def __init__(self, num_factors):
        super().__init__(num_factors)