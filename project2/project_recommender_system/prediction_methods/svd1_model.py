import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from prediction_methods.baseline_model import demean_matrix
from prediction_methods.model_helpers import calculate_rmse
from datafile_methods.data_io import save_csv


def model_svd1(train_data, test_data, test_flag, prediction_path='',
    k=20, n_iter=10, random_state=42, fold_number=''):
    """Matrix factorization by SVD.
    Trains a model on the csr sparse matrix `train_data` and
    creates a prediction for the csr sparse matrix `test_data`.
    If `test_flag` is True, then it also computes train and test rmse.
    The method used is sklearn.decomposition.TruncatedSVD
    """
    assert k <= min(train_data.shape), "k must be smaller than the min dimension of `train_data`"

    def predict(data, filename, save=True):
        # Get non-zero elements
        (rows, cols, vals) = sp.find(data)
        # Do predictions for `data`
        baselines = np.array([(global_mean + item_means[i] + user_means[u])
            for (i, u) in zip(rows, cols)])
        interactions = np.array([(user_features[:,u].dot(item_features[i,:]))
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
    train_dem, global_mean, user_means, item_means = demean_matrix(train_data)

    # Train model using sklearn's TruncatedSVD
    svd = TruncatedSVD(n_components=k, n_iter=n_iter, random_state=random_state)
    item_features = svd.fit_transform(train_dem)
    user_features = svd.components_

    if test_flag:
        # Get predictions for `train_data`
        tr_pred, tr_vals = predict(train_data, '', save=False)
        # Get and save predictions for `test_data`
        te_pred, te_vals = predict(test_data,
            'model_svd1_te_{}'.format(fold_number))
        # Compute train error
        train_rmse = calculate_rmse(tr_vals, tr_pred)
        # Compute test error
        test_rmse = calculate_rmse(te_vals, te_pred)
        return train_rmse, test_rmse
    else:
        # Create and save predictions as Kaggle submissions
        te_pred, te_vals = predict(test_data, 'model_svd1_sub')
