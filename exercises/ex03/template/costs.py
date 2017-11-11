# -*- coding: utf-8 -*-

import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse
    """
    return .5*np.sum(np.square(y-tx.dot(w)))/np.shape(y)[0]

def compute_mae(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mae.
    """
    return .5*np.sum(np.abs(y-tx.dot(w)))/np.shape(y)[0]