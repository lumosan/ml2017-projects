import numpy as np
import scipy.sparse as sp
from prediction_methods.model_helpers import build_index_groups, compute_error
from prediction_methods.baseline_model import demean_matrix, demean_test_matrix
import csv

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
        V = train[i,u].T.dot(M.T)
        A = M.dot(M.T) + n_i_per_user[u] * lambda_u_I
        new_u_features[:,u] = np.linalg.solve(A, V.T).T
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
        V = train[i,u].dot(M.T)
        A = M.dot(M.T) + n_u_per_item[i] * lambda_i_I
        new_i_features[:,i] = np.linalg.solve(A, V.T).T
    return new_i_features


def matrix_factorization_ALS(data, test_data, k=20, lambda_u=.1, lambda_i=.7, tol=1e-6, max_iter=100,
    init_u_features=None, init_i_features=None, sub_filename="new_submission", prediction_path=''):
    """Matrix factorization by ALS"""
    # Set seed
    np.random.seed(988)

    # Substract baseline from data
    data_demeaned, global_mean, user_means, item_means = demean_matrix(data)
    test_demeaned = demean_test_matrix(test_data, global_mean, item_means, user_means)

    # Get non-zero elements
    (rows, cols, vals) = sp.find(data_demeaned)
    (test_rows, test_cols, test_vals) = sp.find(test_demeaned)

    # Initialize feature vectors for users and items
    rand_u_features, rand_i_features = init_MF(data_demeaned, k)
    if init_u_features is None:
        u_features = rand_u_features
    else:
        u_features = init_u_features

    if init_i_features is None:
        i_features = rand_i_features
    else:
        i_features = init_i_features

    # Get number of non-zero ratings per user and item
    n_i_per_user = data_demeaned.getnnz(axis=0)
    n_u_per_item = data_demeaned.getnnz(axis=1)

    # Get non-zero ratings per user and item
    nz_train, nz_u_per_item, nz_i_per_user = build_index_groups(data_demeaned)

    e = 1000

    # ALS-WR algorithm
    for it in range(max_iter):
        u_features = update_user_features(data_demeaned, i_features, lambda_u,
            n_i_per_user, nz_i_per_user)
        i_features = update_item_features(data_demeaned, u_features, lambda_i,
            n_u_per_item, nz_u_per_item)
        # compute and print new training error
        old_e = e
        e = compute_error(data_demeaned, u_features, i_features, nz_train)
        print("training RMSE: {}.".format(e))
        if(abs(old_e - e) < tol):
            print('Finished estimating features')
            break
        if(old_e - e < -tol):
            print('Whoops!')
            break

    ## Do predictions
    #baselines = np.array([(global_mean + item_means[i] + user_means[u])
    #    for (i, u) in zip(test_rows, test_cols)])
    #interactions = np.array([u_features[:,u].dot(i_features[:,i].T)
    #    for (i, u) in zip(test_rows, test_cols)])
    #pred_test = baselines + interactions

    # Compute and print test error
    with open('{dp}{fn}.csv'.format(dp=prediction_path, fn=sub_filename), 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for (i, u) in zip(test_rows, test_cols):
            interaction = u_features[:,u].dot(i_features[:,i].T)
            baseline = global_mean + item_means[i] + user_means[u]
            pred_i_u = interaction + baseline
            writer.writerow({'Id':'r{r}_c{c}'.format(r=i+1,c=u+1),'Prediction':pred_i_u})

    return u_features, i_features

def matrix_factorization_ALS_test(data, test_data,
    k=20, lambda_u=.1, lambda_i=.7, tol=1e-4, max_iter=50,
    init_u_features=None, init_i_features=None):
    """Matrix factorization by ALS"""
    assert k <= min(data.shape), "k must be smaller than the min dimension of 'data'"

    # Demean matrices
    data_demeaned, global_mean, user_means, item_means = demean_matrix(data, verbose=False)
    test_demeaned = demean_test_matrix(test_data, global_mean, item_means, user_means, verbose=False)

    # Get non-zero elements
    (rows, cols, vals) = sp.find(data_demeaned)
    (test_rows, test_cols, test_vals) = sp.find(test_demeaned)

    # Set seed
    np.random.seed(988)

    # Initialize feature vectors for users and items
    rand_u_features, rand_i_features = init_MF(data_demeaned, k)
    if init_u_features is None:
        u_features = rand_u_features
    else:
        u_features = init_u_features

    if init_i_features is None:
        i_features = rand_i_features
    else:
        i_features = init_i_features

    # Get number of non-zero ratings per user and item
    n_i_per_user = data_demeaned.getnnz(axis=0)
    n_u_per_item = data_demeaned.getnnz(axis=1)

    # Get non-zero ratings per user and item
    nz_train, nz_u_per_item, nz_i_per_user = build_index_groups(data_demeaned)

    e = 1000

    # ALS-WR algorithm
    for it in range(max_iter):
        u_features = update_user_features(data_demeaned, i_features, lambda_u,
            n_i_per_user, nz_i_per_user)
        i_features = update_item_features(data_demeaned, u_features, lambda_i,
            n_u_per_item, nz_u_per_item)
        # compute and print new training error
        old_e = e
        e = compute_error(data_demeaned, u_features, i_features, nz_train)
        print("training RMSE: {}.".format(e))
        if(abs(old_e - e) < tol):
            print('Finished estimating features')
            break
        if(old_e - e < -2*tol):
            print('Whoops!')
            break

    ## Do predictions
    #baselines = np.array([(global_mean + item_means[i] + user_means[u])
    #    for (i, u) in zip(test_rows, test_cols)])
    #interactions = np.array([u_features[:,u].dot(i_features[:,i].T)
    #    for (i, u) in zip(test_rows, test_cols)])
    #pred_test = baselines + interactions

    # Compute and print test error
    nnz_row, nnz_col = test_demeaned.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    rmse = compute_error(test_demeaned, u_features, i_features, nnz_test)
    print("test RMSE after running ALS: {v}.".format(v=rmse))

    return u_features, i_features
