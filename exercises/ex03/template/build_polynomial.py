# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    '''Polynomial basis functions for input data x, for up to a certain degree.'''
    rows, cols = np.indices((x.shape[0], degree + 1))
    return np.power(x[rows], cols)