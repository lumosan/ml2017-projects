# Auxiliary functions
import numpy as np

# Functions for preparing data
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x

def build_poly(x, degree, offset=True):
    """Polynomial basis functions for input data x,
    for up to a certain degree."""
    if len(np.array(x).shape) == 1:
        if offset:
            rows, cols = np.indices((x.shape[0], degree+1))
            tx = np.power(x[rows], cols)
        else:
            rows, cols = np.indices((x.shape[0], degree))
            tx = np.power(x[rows], cols+1)
    else:
        rows, cols = np.indices((x.shape[0], degree))
        xT = np.array(x).T
        tx = np.ones([x.shape[0],1])
        for r in xT:
            tx_r = np.power(r[rows], cols+1)
            tx = np.concatenate([tx, tx_r], axis=1)
        if not offset:
            tx = tx[:,1:]
    return tx

def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    tx = np.c_[np.ones(len(y)), x]
    return y, tx

def split_data(x, y, ratio, seed=1):
    """Split the dataset based on the split ratio.

    E.g. if the ratio is 0.8 then 80% of the data set is dedicated
    to training (and the rest dedicated to testing)
    """
    np.random.seed(seed)

    size = int(ratio * x.shape[0])
    indices = np.random.permutation(x.shape[0])
    training_idx, test_idx = indices[:size], indices[size:]

    x_training, x_test = x[training_idx], x[test_idx]
    y_training, y_test = y[training_idx], y[test_idx]

    return x_training, x_test, y_training, y_test

# Cost and gradient computation functions
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

def compute_mae(y, tx, w):
    """Calculates the loss using mae."""
    return np.sum(np.abs(y - tx.dot(w))) / np.shape(y)[0]

def logistic_by_gd(y, tx, w, gamma):
    """Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    tx_w = tx.dot(w)
    # compute the cost
    loss = np.sum(np.log(1 + np.exp(tx_w))) - np.sum(y * tx_w)
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
        loss = (np.sum(np.log(1 + np.exp(tx_w))) - np.sum(y * tx_w) +
            lambda_ / 2 * compute_mse(y, tx, w))
        # compute the gradient
        gradient = tx.T.dot(sigmoid(tx_w) - y) + lambda_ * w
    else:
        w_no_offset=w
        w_no_offset[0]=0
        loss = (np.sum(np.log(1 + np.exp(tx_w))) - np.sum(y * tx_w) +
            lambda_ / 2 * compute_mse(y, tx, w_no_offset))
        # compute the gradient
        gradient = tx.T.dot(sigmoid(tx_w) - y) + lambda_ * w_no_offset
    # update w
    w_1 = w
    w = w_1 - gamma * gradient
    return loss, w

# Other functions
def sigmoid(t):
    """Apply sigmoid function on t, with t being a one dim vector"""
    t_exp = np.exp(t)
    return t_exp / (t_exp + 1)

def get_batch(y, tx, batch_size):
    m = tx.shape[0]
    p = np.random.permutation(np.arange(m))
    tx_p, y_p = tx[p], y[p]
    return y_p[:batch_size], tx_p[:batch_size,:]

def adaptive_gamma(kappa=0.75, eta0=1e-5):
    t = 1
    while True:
        yield eta0*t**-kappa
        t += 1