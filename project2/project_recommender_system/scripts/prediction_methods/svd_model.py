import numpy as np
import scipy.sparse as sp
import csv
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from prediction_methods.baseline_model import demean_matrix
from prediction_methods.model_helpers import calculate_mse

def matrix_factorization_SVD(data, test_data, test_flag, sub_flag=False,
    k=20, int_vals=False, sub_filename="new_submission", verbose=False,
    n_iter=10, random_state=42, library='scipy', prediction_path=''):
    """Matrix factorization by SVD.

    If 'test_flag' is True, then 'data' should be the training dataset and
    'test_data' the test dataset. In this case sub_flag is ignored.

    If 'test_flag' is False and 'sub_flag' is True, then 'data' should be
    the entire ratings dataset and 'test_data' should be a sample submission.

    Both 'data' and 'test_data' should be csr sparse matrices.
    """

    assert test_flag or sub_flag, "Specify a task"
    assert k <= min(data.shape), "k must be smaller than the min dimension of 'data'"

    # Set seed
    np.random.seed(988)

    # Substract baseline from data
    r_demeaned, global_mean, user_means, item_means = demean_matrix(data, verbose=verbose)

    if library == 'scipy':
        # Use scipy's svds
        U, sigma, Vt = svds(r_demeaned, k)
        sigma = np.diag(sigma)
        U_sigma = np.dot(U, sigma)
    else:
        # Use sklearn's TruncatedSVD
        svd = TruncatedSVD(n_components=k, n_iter=n_iter, random_state=random_state)
        item_features = svd.fit_transform(r_demeaned)
        user_features = svd.components_
    if verbose:
        print('Finished fitting model')

    # Get non-zero elements
    (rows, cols, vals) = sp.find(data)
    (test_rows, test_cols, test_vals) = sp.find(test_data)

    if test_flag:
        # Do predictions
        baselines = np.array([(global_mean + item_means[i] + user_means[u])
            for (i, u) in zip(test_rows, test_cols)])
        if library == 'scipy':
            interactions = np.array([(U_sigma[i,:].dot(Vt[:,u]))
                for (i, u) in zip(test_rows, test_cols)])
        else:
            interactions = np.array([(user_features[:,u].dot(item_features[i,:]))
                for (i, u) in zip(test_rows, test_cols)])
        pred_test = baselines + interactions
        if int_vals:
            pred_test = np.rint(pred_test)

        if verbose:
            print('Finished predicting')

        # Compute and print test error
        test_mse = calculate_mse(test_vals, pred_test)
        test_rmse = np.sqrt(test_mse / len(test_vals))

        if verbose:
            print("Test RMSE using baseline and matrix factorization: {e}".format(e=test_rmse))

        return test_rmse, pred_test

    elif sub_flag:
        # Directly write predictions to submission file
        with open('{dp}{fn}.csv'.format(dp=prediction_path, fn=sub_filename), 'w') as csvfile:
            fieldnames = ['Id', 'Prediction']
            writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
            writer.writeheader()
            for (i, u) in zip(test_rows, test_cols):
                if library == 'scipy':
                    interaction = U_sigma[i,:].dot(Vt[:,u])
                else:
                    interaction = user_features[:,u].dot(item_features[i,:])
                baseline = global_mean + user_means[u] + item_means[i]
                pred_i_u = interaction + baseline
                writer.writerow({'Id':'r{r}_c{c}'.format(r=i+1,c=u+1),'Prediction':pred_i_u})

        if verbose:
            print('Completed submission in matrix_factorization_SVD!')
