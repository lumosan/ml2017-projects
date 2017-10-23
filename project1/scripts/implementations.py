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
        gradient = (tx.T.dot(tx.dot(w) - y)) / len(y)
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
    random.seed(seed)
    w = initial_w
    if return_all:
        ws = [w]
        losses = []
    for n_iter in range(max_iters):
        # compute gradient and loss
        w_1 = w
        n = np.random.randint(0, np.shape(y)[0])
        grad_aux = y[n] - tx[n].dot(w)
        gradient = -(tx[n].dot(grad_aux))

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

def logistic_regression(y, tx, initial_w, max_iters, gamma, return_all=False):
    if len(tx.shape) == 1:
        tx = tx.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    initial_w = np.array(initial_w).reshape(tx.shape[1], 1)

    # init parameters
    w = initial_w
    if return_all:
        ws = [w]
        losses = []

    for n_iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        if return_all:
            ws.append(w)
            losses.append(loss)
            if n_iter % 100 == 0:
                print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))

    if return_all:
        return losses, ws
    else:
        return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma,
    return_all=False):

    if len(tx.shape) == 1:
        tx = tx.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    initial_w = np.array(initial_w).reshape(tx.shape[1], 1)

    # init parameters
    w = initial_w
    if return_all:
        ws = [w]
        losses = []

    for n_iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_reg_gradient_descent(y, tx, w, gamma)
        if return_all:
            ws.append(w)
            losses.append(loss)
            if n_iter % 100 == 0:
                print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))

    if return_all:
        return losses, ws
    else:
        return loss, w