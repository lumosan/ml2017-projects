# Functions for cross-validation
import numpy as np


def calculate_precision(y, y_pred, zeros_ones=False):
    """Calculates precision by comparing the observed labels
    and the predicted ones. Assumes labels used are -1 and 1
    unless zeros_ones is set to True.
    The precision is defined as the ratio of correct labels.
    """
    if len(np.array(y).shape) > 1:
        y=y.flatten()
    if len(np.array(y_pred).shape) > 1:
        y_pred=y_pred.flatten()
    if zeros_ones:
        incorrect = np.sum(np.abs(y - y_pred))
    else:
        incorrect = np.sum(np.abs(y - y_pred))/2
    precision = 1 - (incorrect / y.shape[0])
    return precision

def split_data(x, y, ratio, seed=1):
    """Splits the dataset based on the split ratio.

    E.g. if the ratio is 0.8 then 80% of the data set is dedicated
    to training (and the rest dedicated to testing)
    """
    # set random seed
    np.random.seed(seed)
    # generate indexes for splitting
    size = int(ratio * x.shape[0])
    indices = np.random.permutation(x.shape[0])
    training_idx, test_idx = indices[:size], indices[size:]
    # data splitting
    x_training, x_test = x[training_idx], x[test_idx]
    y_training, y_test = y[training_idx], y[test_idx]

    return x_training, x_test, y_training, y_test
