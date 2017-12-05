import numpy as np
import scipy.sparse as sp
from prediction_methods.model_helpers import compute_division



# Baseline rating
def baseline_rating(data):
    """Implements baseline method for a ratings matrix
    using the global mean.
    """
    # Compute global mean using training data
    r_mean = data.sum() / data.getnnz()
    return r_mean

# User or item specific effect
def baseline_user_item_specific(data, mean, set_num=0):
    """Implements baseline method for a ratings matrix
    using either the user or the item mean,
    as indicated in parameter mean.
    """
    if mean=="user":
        flag = 1
        inv_flag = 0
    else:
        flag = 0
        inv_flag = 1

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

    # Substract the baseline of each element in 'data'
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
    """Removes the global, user and item means from a matrix
    using the known means. Returns the demeaned matrix.
    """
    num_items, num_users = data.shape
    (rows, cols, vals) = sp.find(data)

    # Substract the baseline of each element in 'data'
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



def model_baseline(data, test_data, test_flag, sub_flag=False,
    sub_filename="new_submission", verbose=False):

    """If 'test_flag' is True, then 'data' should be the training dataset
    'test_data' the test dataset. In this case sub_flag is ignored.

    If 'test_flag' is False and 'sub_flag' is True, then 'data' should be
    the entire ratings dataset and 'test_data' should be a sample submission.

    Both 'data' and 'test_data' should be csr sparse matrices.
    """
    assert test_flag or sub_flag, "Specify a task"

    num_train_items, num_train_users = data.shape
    num_test_items, num_test_users = test_data.shape

    num_i_max = max(num_train_items, num_test_items)
    num_u_max = max(num_train_users, num_test_users)

    global_mean = baseline_rating(data)
    item_means = baseline_user_item_specific(data, 'item', set_num=num_i_max)
    user_means = baseline_user_item_specific(data, 'user', set_num=num_u_max)

    (rows, cols, vals) = sp.find(test_data)

    if test_flag:
        # Do predictions
        pred_test = np.array([(global_mean + item_means[i] + user_means[u])
            for (i, u) in zip(rows, cols)])

        # Compute and print test error
        test_mse = calculate_mse(vals, pred_test)
        test_rmse = np.sqrt(test_mse / len(vals))
        if verbose:
            print("Test RMSE of baseline using baseline: {e}".format(e=test_rmse))
        return test_rmse, pred_test

    elif sub_flag:
        # Directly write predictions to submission file
        with open('{dp}{fn}.csv'.format(dp=PREDICTION_PATH, fn=sub_filename), 'w') as csvfile:
            fieldnames = ['Id', 'Prediction']
            writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
            writer.writeheader()
            for (i, u) in zip(rows, cols):
                pred_i_u = global_mean + user_means[u] + item_means[i]
                writer.writerow({'Id':'r{r}_c{c}'.format(r=i+1,c=u+1),'Prediction':pred_i_u})