# Auxiliary functions
import numpy as np


# Standardizing functions

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




# Data processing functions

def build_poly(x, degree, offset=True):
    """Polynomial basis functions for input data x,
    for up to a certain degree.
    """
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

def build_model_data(x):
    """Forms tX to get regression data in matrix form.
    It appends a column of ones to x.
    """
    tx = np.c_[np.ones(x.shape[0]), x]
    return tx

def PCA_analysis(x, k, transform_test=False, test_x=0):
    """Applies eigendecomposition to x in order to return a transformed
    tx with the k eigenvectors linked to the k largest eigenvalues.
    """
    # compute covariance matrix
    cov_mat = np.cov(x.T)
    # compute eigenvalues and eigenvectors
    e_val, e_vec = np.linalg.eigh(cov_mat)
    # take 10 eigenvectors with highest eigenvalues
    ind = np.argpartition(e_val, -k)[-k:]
    e_val_ind, e_vec_ind = e_val[ind], e_vec[ind]
    # transform samples onto the new subspace
    transformed = x.dot(e_vec_ind.T)
    if transform_test:
        test_transformed = test_x.dot(e_vec_ind.T)
        return transformed, test_transformed
    else:
        return transformed

def build_poly_PCA(input_data, test_input_data, degree,
    num_comp, apply_pca=True):
    """Generates a matrix tx based on input_data, by first building the
    polynomial basis of each column of input_data and then applying PCA
    to keep only num_comp variables.
    It returns the normalized tx with an offset vector.
    """
    # build poly
    poly_x = build_poly(input_data, degree, offset=False)
    test_poly_x = build_poly(test_input_data, degree, offset=False)
    # standardize
    x, mean_x, std_x = standardize(poly_x)
    test_x = standardize_test(test_poly_x, mean_x, std_x)

    if apply_pca:
        # apply PCA
        transf_x, test_transf_x = PCA_analysis(x, num_comp,
            transform_test=True, test_x=test_x)
        # add offset
        poly_tx = build_model_data(transf_x)
        test_poly_tx = build_model_data(test_transf_x)
    else:
        # add offset
        poly_tx = build_model_data(x)
        test_poly_tx = build_model_data(test_x)

    return poly_tx, test_poly_tx




# Functions for cross-validation

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

def predict_labels_bis(weights, data, return_zeros=False):
    """Generates class predictions given weights and a test data matrix.
    It asummes that the model generates the prediction using classes
    0 and 1. It can return labels 0 and 1, or -1 and 1 depending on the
    value of return_zeros.
    """
    y_pred = np.dot(data, weights)
    if return_zeros:
        y_pred[np.where(y_pred <= 0.5)] = 0
    else:
        y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred




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




# Other functions

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