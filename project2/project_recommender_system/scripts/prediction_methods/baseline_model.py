import numpy as np
import scipy.sparse as sp
from datafile_methods.data_io import save_csv
from prediction_methods.model_helpers import compute_division, calculate_mse


def baseline_rating(data):
    """Implements baseline method for a ratings matrix using
    the global mean.
    """
    # Compute global mean using training data
    r_mean = data.sum() / data.getnnz()
    return r_mean


def baseline_user_item_specific(data, mean, set_num=0):
    """Implements baseline method for a ratings matrix using either the
    user or the item mean, as indicated in parameter mean.
    Returns the user or item specific effect.
    """
    if mean=="user":
        flag = 1
        inv_flag = 0
    else:
        flag = 0
        inv_flag = 1

    # Number of elements for which the baseline is computed
    num = max(set_num, data.shape[flag])

    # Obtain r_demeaned (ratings minus global avg)
    global_mean = baseline_rating(data)
    r_demeaned = data.copy()
    r_demeaned.data = (1.0 * r_demeaned.data) - global_mean

    # Compute means using training data
    # get rows, columns and values for elements in r_demeaned
    data_rcv = sp.find(r_demeaned)
    # compute means
    counts = np.bincount(data_rcv[flag], minlength=num)
    sums = np.bincount(data_rcv[flag], weights=data_rcv[2], minlength=num)
    means = compute_division(sums, counts)
    return means


def demean_matrix(data, verbose=False):
    """Removes the global, user and item means from a matrix.
    Returns the matrix and the computed means.
    """
    num_rows, num_cols = data.shape
    (rows, cols, vals) = sp.find(data)

    # Compute global, user and item means
    global_mean = baseline_rating(data)
    item_means = baseline_user_item_specific(data, 'item')
    user_means = baseline_user_item_specific(data, 'user')

    # Substract baseline of each element in `data`
    train_vals = vals.copy()
    train_vals = 1.0 * train_vals

    baselines = np.array([(global_mean + item_means[i] + user_means[u])
        for (i, u) in zip(rows, cols)])
    train_vals = train_vals - baselines

    # Get matrix
    r_demeaned = sp.csr_matrix((train_vals, (rows, cols)),
        shape=(num_rows, num_cols))
    if verbose:
        print('Completed demean_matrix!')
    return r_demeaned, global_mean, user_means, item_means


def demean_test_matrix(data, global_mean, item_means, user_means,
    verbose=False):
    """Removes known global, user and item means from a matrix.
    Returns the demeaned matrix.
    """
    num_items, num_users = data.shape
    (rows, cols, vals) = sp.find(data)

    # Substract baseline of each element in `data`
    train_vals = vals.copy()
    train_vals = 1.0 * train_vals
    baselines = np.array([(global_mean + item_means[i] + user_means[u])
        for (i, u) in zip(rows, cols)])
    train_vals -= baselines

    # Get matrix
    r_demeaned = sp.csr_matrix((train_vals, (rows, cols)),
        shape=(num_items, num_users))

    if verbose:
        print('Completed demean_matrix!')
    return r_demeaned


def model_baseline(train_data, test_data, test_flag, prediction_path=''):
    """Baseline by global, item and user mean
    Trains a model on the csr sparse matrix `train_data` and
    creates a prediction for the csr sparse matrix `test_data`.
    If `test_flag` is True, then it also computes train and test rmse.
    """
    def predict(data, filename, save=True):
        # Get non-zero elements
        (rows, cols, vals) = sp.find(data)
        # Do predictions for `data`
        pred = np.array([(global_mean + item_means[i] + user_means[u])
            for (i, u) in zip(rows, cols)])
        pred = np.clip(pred, 1.0, 5.0)
        if save:
            # Write predictions to submission file
            pred_matrix = sp.csr_matrix((pred, (rows, cols)), shape=data.shape)
            save_csv(pred_matrix, prediction_path=prediction_path,
                filename=filename)
        return pred, vals

    # Obtain number of items and users
    num_tr_i, num_tr_u = train_data.shape
    num_te_i, num_te_u = test_data.shape
    num_i_max = max(num_tr_i, num_te_i)
    num_u_max = max(num_tr_u, num_te_u)

    # Obtain global, item and user means baselines
    global_mean = baseline_rating(train_data)
    item_means = baseline_user_item_specific(train_data, 'item', set_num=num_i_max)
    user_means = baseline_user_item_specific(train_data, 'user', set_num=num_u_max)

    if test_flag:
        # Get predictions for `train_data`
        tr_pred, tr_vals = predict(train_data, '', save=False)
        # Get and save predictions for `test_data`
        te_pred, te_vals = predict(test_data, '', save=False)
        # Compute train error
        train_mse = calculate_mse(tr_vals, tr_pred)
        train_rmse = np.sqrt(train_mse / len(tr_vals))
        # Compute test error
        test_mse = calculate_mse(te_vals, te_pred)
        test_rmse = np.sqrt(test_mse / len(te_vals))
        return train_rmse, test_rmse
    else:
        # Create and save predictions as Kaggle submissions
        te_pred, te_vals = predict(test_data, 'model_baseline_sub')
        tr_pred, tr_vals = predict(train_data, 'model_baseline_tr')
