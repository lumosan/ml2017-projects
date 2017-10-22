# Implementations of the functions from exercise sessions
import numpy as np
from auxiliary_functions import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma, return_all=False):
    """Linear regression using gradient descent"""
    w = initial_w
    if return_all:
        ws = [w]
        losses = []
    for n_iter in range(max_iters):
        w_1 = w
        gradient = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w_1 - gamma * gradient

        if return_all:
            ws.append(w)
            losses.append(loss)
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    if return_all:
        return losses, ws
    else:
        return loss, w

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma,
    seed=13, return_all=False):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    if return_all:
        ws = [w]
        losses = []
    for n_iter in range(max_iters):
        # compute gradient and loss
        w_1 = w
        gradient = compute_stoch_gradient(y, tx, w, seed)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        w = w_1 - gamma * gradient

        if return_all:
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    if return_all:
        return losses, ws
    else:
        return loss, w

def least_squares(y, tx):
    """Least squares regression using normal equations.
    It returns RMSE and the optimal weights.
    """
    w_opt = np.linalg.pinv(tx).dot(y)
    rmse = np.sqrt(compute_mse(y, tx, w_opt))
    return w_opt, rmse

# TODO: Differentiate cases when tx includes offset column and when it doesn't?
def ridge_regression(y, tx, lambda_):
    """Implements ridge regression using normal equations."""
    I = np.eye(tx.shape[1])
    inv = np.linalg.inv(tx.T.dot(tx) + 2 * tx.shape[1] * lambda_ * I)
    w_opt = inv.dot(tx.T).dot(y)
    rmse = np.sqrt(compute_mse(y, tx, w_opt))
    return w_opt, rmse

