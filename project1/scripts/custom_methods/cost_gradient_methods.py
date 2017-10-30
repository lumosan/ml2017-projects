# Cost and gradient computation functions
import numpy as np
from custom_methods.auxiliary_methods import *


def compute_mse(y, tx, w):
    """Calculates the loss using MSE."""
    if len(tx.shape) == 1:
        tx = tx.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    w = np.array(w).reshape(tx.shape[1], 1)
    z = y - tx.dot(w)
    z = z.T.dot(z)
    return z[0][0] / tx.shape[0]

def logistic_by_gd(y, tx, w, gamma):
    """Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    tx_w = tx.dot(w)
    # compute the cost
    loss = np.sum(np.logaddexp(0, tx_w)) - np.sum(y * tx_w)
    # compute the gradient
    gradient = tx.T.dot(sigmoid(tx_w) - y)
    # update w
    w_1 = w
    w = w_1 - gamma * gradient
    return loss, w

def reg_logistic_by_gd(y, tx, lambda_, w, gamma, penalize_offset):
    """Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    tx_w = tx.dot(w)
    if penalize_offset:
        # compute the cost
        loss = (np.sum(np.logaddexp(0, tx_w)) - np.sum(y * tx_w) +
            lambda_ / 2 * compute_mse(y, tx, w))
        # compute the gradient
        gradient = tx.T.dot(sigmoid(tx_w) - y) + lambda_ * w
    else:
        w_no_offset=w
        w_no_offset[0]=0
        loss = (np.sum(np.logaddexp(0, tx_w)) - np.sum(y * tx_w) +
            lambda_ / 2 * compute_mse(y, tx, w_no_offset))
        # compute the gradient
        gradient = tx.T.dot(sigmoid(tx_w) - y) + lambda_ * w_no_offset
    # update w
    w_1 = w
    w = w_1 - gamma * gradient
    return loss, w
