# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
import numpy as np
from costs import compute_loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    n = np.random.randint(0, np.shape(y)[0])
    aux1 = y[n] - tx[n].dot(w)
    return -(tx[n].dot(aux1))


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_stoch_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        aux = w - gamma * gradient
        w = aux
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws