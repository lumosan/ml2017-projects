import numpy as np
import scipy.sparse as sp
from recommend.als import ALS
from recommend.utils.evaluation import RMSE
from numpy.random import RandomState
from datafile_methods.data_io import save_csv_rec


def model_mf_als_recommend(train_data, test_data, test_flag, n_user=5, n_item=5,
    prediction_path='', validation_data=None, k=30, n_iter=50, reg=5e-2, seed=0):
    """Matrix factorization by ALS using the library recommend.
    Trains a model on the csr sparse matrix `train_data` and
    creates a prediction for the csr sparse matrix `test_data`.
    If `test_flag` is True, then it also computes train and test rmse.
    """
    def predict(data, filename, save=True):
        # Get non-zero elements
        (rows, cols, vals) = sp.find(data)
        # Create `ratings` array
        ratings = np.array([np.array([int(u),int(i),int(r)])
            for (u,i,r) in zip(cols, rows, vals)])
        # Do predictions for `ratings`
        pred = als.predict(ratings[:, :2])
        if save:
            # Write predictions to submission file
            save_csv_rec(ratings, pred, prediction_path=prediction_path,
                filename=filename)
        return pred, ratings[:, 2]

    # Set seed and RandomState
    # TODO: Not sure I need them...
    np.random.seed(0)
    rand_state = RandomState(0)

    # Initialize constants
    max_rating, min_rating = 5.0, 1.0

    # Get non-zero values in `train_data` and create `tr_ratings` array
    (tr_rows, tr_cols, tr_vals) = sp.find(train_data)
    tr_ratings = np.array([np.array([int(u),int(i),int(r)])
        for (u,i,r) in zip(tr_cols, tr_rows, tr_vals)])

    # Create and train model
    als = ALS(n_user=n_user, n_item=n_item, n_feature=k, reg=reg,
        max_rating=max_rating, min_rating=min_rating, seed=seed)
    als.fit(tr_ratings, n_iters=n_iter)

    if test_flag:
        # Get predictions for `train_data`
        tr_pred, tr_vals = predict(train_data, '', save=False)
        # Get and save predictions for `test_data`
        te_pred, te_vals = predict(test_data, '', save=False)
        # Compute train error
        train_rmse = RMSE(tr_pred, tr_vals)
        # Compute test error
        test_rmse = RMSE(te_pred, te_vals)
        return train_rmse, test_rmse
    else:
        # Create and save predictions as Kaggle submissions
        te_pred, te_vals = predict(test_data, 'model_mf_als_recommend_sub')
        tr_pred, tr_vals = predict(train_data, 'model_mf_als_recommend_tr')
