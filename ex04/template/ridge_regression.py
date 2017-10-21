# -*- coding: utf-8 -*-
import numpy as np

def compute_mse(y, tx, w):
    '''
    Calculates the loss using MSE.
    '''
    if len(tx.shape) == 1:
        tx = tx.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    w = np.array(w).reshape(tx.shape[1], 1)
    z = y - tx.dot(w)
    z = z.T.dot(z)
    return z[0][0] / tx.shape[0]

def ridge_regression(y, tx, lambda_):
    '''
    Implements ridge regression using normal equations.
    '''
    I = np.eye(tx.shape[1])
    inv = np.linalg.inv(tx.T.dot(tx) + 2 * tx.shape[1] * lambda_ * I)
    w_opt = inv.dot(tx.T).dot(y)
    rmse = np.sqrt(compute_mse(y, tx, w_opt))
    return w_opt, rmse
