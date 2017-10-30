# Data processing functions
import numpy as np
from standardizing_methods import standardize, standardize_test


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