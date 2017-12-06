import numpy as np
import scipy.sparse as sp
from recommend.als import ALS
from recommend.utils.evaluation import RMSE
from numpy.random import RandomState
from processing_methods.data_processing import save_csv_rec


def model_mf_als_recommend(train_data, test_data, test_flag, n_user, n_item,
    prediction_path='', validation_data=None, k=30, n_iter=50, reg=5e-2, seed=0):
    """Matrix factorization by ALS using the library recommend.
    Trains a model on the csr sparse matrix `train_data` and
    creates a prediction for the csr sparse matrix `test_data`.
    If `test_flag` is True, then it also computes rmse for `test_data`
    and creates predictions for `validation_data`.
    """
    def predict(data, header, filename):
        # Get non-zero elements
        (rows, cols, vals) = sp.find(data)
        # Create `ratings` array
        ratings = np.array([np.array([int(u),int(i),int(r)])
            for (u,i,r) in zip(cols, rows, vals)])
        # Do predictions for `ratings`
        pred = als.predict(ratings[:, :2])
        # Write predictions to submission file
        save_csv_rec(ratings, pred, header=header, prediction_path=prediction_path,
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
        # Do and write predictions for `test_data` and `validation_data`
        te_pred, te_vals = predict(test_data, False, 'model_mf_als_recommend_te')
        val_pred, val_vals = predict(validation_data, False, 'model_mf_als_recommend_val')

        # Compute and print error for `test_data`
        test_rmse = RMSE(te_pred, te_vals)
        print("Test RMSE of model_mf_als_recommend: {e}".format(e=test_rmse))
    else:
        # Create prediction for `test_data` and save it as a Kaggle submission
        te_pred, te_vals = predict(test_data, True, 'model_mf_als_recommend_sub')
