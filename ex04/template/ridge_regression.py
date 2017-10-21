# -*- coding: utf-8 -*-
import numpy as np


def ridge_regression(y, tx, lambda_):
    '''
    Implements ridge regression using normal equations.
    '''
    I = np.eye(tx.shape[1])
    inv = np.linalg.inv(tx.T.dot(tx) + 2 * tx.shape[1] * lambda_ * I)
    w_opt = inv.dot(tx.T).dot(y)
    #rmse = np.sqrt(compute_mse(y, tx, w_opt) +
    #    lambda_ * np.linalg.norm(w_opt)**2)
    rmse = np.sqrt(compute_mse(y, tx, w_opt))
    return w_opt, rmse
