# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np


def compute_mse(y, tx, w):
    """Calculates the loss using mse."""
    if len(tx.shape) == 1:
        tx = tx.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    w = np.array(w).reshape(tx.shape[1], 1)
    z = y - tx.dot(w)
    z = z.T.dot(z)
    return z[0][0] / tx.shape[0]

def compute_mae(y, tx, w):
    """Calculates the loss using mae."""
    return np.sum(np.abs(y - tx.dot(w))) / np.shape(y)[0]

def compute_loss(y, tx, w):
    """Calculates the loss using mse."""
    return compute_mse(y, tx, w)