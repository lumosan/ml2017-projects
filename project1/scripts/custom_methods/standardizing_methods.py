# Standardizing functions
import numpy as np



def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

def standardize_test(x, means, stds):
    """Standardize data set with given mean and std.
    Used so that the test dataset is transformed in
    the same way as the training dataset.
    """
    x_1 = x - means
    x_2 = x_1 / stds
    return x_2

def standardize_ignoring_values(x, nan):
    """Standardize a dataset ignoring certain values.
    The ignored values are set to 0 after standardizing.
    """
    cs=[]
    means=[]
    stds=[]
    for c in x.T:
        # standardize
        mean = np.mean(c[np.where(c!=nan)])
        c_1 = c - mean
        std = np.std(c_1[np.where(c!=nan)])
        c_2 = c_1 / std
        # set nan values to 0
        c_2[np.where(c==nan)]=0
        means.append(mean)
        stds.append(std)
        cs.append(c_2)
    return np.array(cs).T, np.array(means), np.array(stds)

def standardize_test_ignoring_values(x, nan, means, stds):
    """Standardize a dataset ignoring certain values.
    The ignored values are set to 0 after standardizing.
    Used so that the test dataset is transformed in
    the same way as the training dataset.
    """
    # standardize
    x_1 = x - means
    x_2 = x_1 / stds
    # set nan values to zero
    x_2[np.where(x==nan)]=0
    return x_2

