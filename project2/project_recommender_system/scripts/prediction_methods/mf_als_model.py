import numpy as np
import scipy.sparse as sp
from prediction_methods.model_helpers import build_index_groups, compute_error, calculate_mse
from prediction_methods.baseline_model import demean_matrix, demean_test_matrix
from datafile_methods.data_io import save_csv


def init_MF(data, k):
    """Initializes parameters for Matrix Factorization.
    Assumes 'data' matrix is already demeaned.
    """
    np.random.seed(988)
    num_items, num_users = data.shape
    u_features = np.random.rand(k, num_users)
    i_features = np.random.rand(k, num_items)
    return u_features, i_features


def update_user_features(train, i_features, lambda_u,
    n_i_per_user, nz_i_per_user):
    """Updates user feature matrix."""
    n_u = len(nz_i_per_user)
    k = i_features.shape[0]
    lambda_u_I = lambda_u * sp.eye(k)
    new_u_features = np.zeros((k, n_u))
    for u, i in nz_i_per_user:
        M = i_features[:,i]
        V = M.dot(train[i,u])
        A = M.dot(M.T) + n_i_per_user[u] * lambda_u_I
        X = np.linalg.solve(A, V)
        new_u_features[:,u] = X.T
    return new_u_features


def update_item_features(train, u_features, lambda_i,
    n_u_per_item, nz_u_per_item):
    """Updates item feature matrix."""
    n_i = len(nz_u_per_item)
    k = u_features.shape[0]
    lambda_i_I = lambda_i * sp.eye(k)
    new_i_features = np.zeros((k, n_i))
    for i, u in nz_u_per_item:
        M = u_features[:,u]
        V = M.dot(train[i,u].T)
        A = M.dot(M.T) + n_u_per_item[i] * lambda_i_I
        X = np.linalg.solve(A, V)
        new_i_features[:,i] = X.T
    return new_i_features


def model_mf_als(train_data, test_data, test_flag, prediction_path='',
    k=20, lambda_u=.1, lambda_i=.7, tol=1e-6, max_iter=100,
    init_u_features=None, init_i_features=None):
    """Matrix factorization by ALS
    Trains a model on the csr sparse matrix `train_data` and
    creates a prediction for the csr sparse matrix `test_data`.
    If `test_flag` is True, then it also computes train and test rmse.
    """
    def predict(data, filename, save=True):
        # Get non-zero elements
        (rows, cols, vals) = sp.find(data)
        # Do predictions for `data`
        baselines = np.array([(global_mean + item_means[i] + user_means[u])
            for (i, u) in zip(rows, cols)])
        interactions = np.array([u_features[:,u].dot(i_features[:,i].T)
            for (i, u) in zip(rows, cols)])
        pred = baselines + interactions
        pred = np.clip(pred, 1.0, 5.0)
        if save:
            # Write predictions to submission file
            pred_matrix = sp.csr_matrix((pred, (rows, cols)), shape=data.shape)
            save_csv(pred_matrix, prediction_path=prediction_path,
                filename=filename)
        return pred, vals

    # Set seed
    np.random.seed(988)

    # Substract baseline from `train_data`
    train_dem_csr, global_mean, user_means, item_means = demean_matrix(train_data)
    train_dem = train_dem_csr.todense()

    # Initialize feature vectors for users and items
    rand_u_features, rand_i_features = init_MF(train_dem_csr, k)
    if init_u_features is None:
        u_features = rand_u_features
    else:
        u_features = init_u_features

    if init_i_features is None:
        i_features = rand_i_features
    else:
        i_features = init_i_features

    # Get number of non-zero ratings per user and item
    n_i_per_user = train_dem_csr.getnnz(axis=0)
    n_u_per_item = train_dem_csr.getnnz(axis=1)

    # Get non-zero ratings per user and item
    nz_train, nz_u_per_item, nz_i_per_user = build_index_groups(train_dem_csr)

    e = 1000

    # ALS-WR algorithm
    for it in range(max_iter):
        u_features = update_user_features(train_dem, i_features, lambda_u,
            n_i_per_user, nz_i_per_user)
        i_features = update_item_features(train_dem, u_features, lambda_i,
            n_u_per_item, nz_u_per_item)
        # compute and print new training error
        old_e = e
        e = compute_error(train_dem_csr, u_features, i_features, nz_train)
        if(abs(old_e - e) < tol):
            break
        if(old_e - e < -tol):
            #TODO: Remove this print and ask a TA about this
            break

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
        te_pred, te_vals = predict(test_data, 'model_mf_als_sub')
        tr_pred, tr_vals = predict(train_data, 'model_mf_als_tr')
