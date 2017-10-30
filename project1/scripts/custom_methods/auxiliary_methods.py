# Other functions
import numpy as np

def sigmoid(t):
    """Apply sigmoid function on t, with t being a one dim vector"""
    return 1 / (1 + np.exp(-t))

def get_batch(y, tx, batch_size):
    """Generates a batch of size batch_size.
    Used for Stochastic Gradient Descent.
    """
    m = tx.shape[0]
    p = np.random.permutation(np.arange(m))
    tx_p, y_p = tx[p], y[p]
    return y_p[:batch_size], tx_p[:batch_size,:]

def adaptive_gamma(kappa=0.75, eta0=1e-5):
    """Adaptive learning rate. After creating the gamma with the
    values for kappa and eta0, it yields the value for the learning
    rate of the next iteration. Used for (Stochastic) Gradient Descent
    methods.
    """
    t = 1
    while True:
        yield eta0*t**-kappa
        t += 1