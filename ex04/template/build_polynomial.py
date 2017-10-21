# -*- coding: utf-8 -*-
import numpy as np

def build_poly(x, degree, offset=True):
    '''
    Polynomial basis functions for input data x,
    for up to a certain degree.
    '''
    if offset:
        rows, cols = np.indices((x.shape[0], degree+1))
        tx = np.power(x[rows], cols)
    else:
        rows, cols = np.indices((x.shape[0], degree))
        tx = np.power(x[rows], cols+1)
    return tx
