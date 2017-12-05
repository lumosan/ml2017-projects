import numpy as np
import scipy.sparse as sp
from recommend.als import ALS
from recommend.utils.evaluation import RMSE
from numpy.random import RandomState
import csv


def recommend_ALS_test(data, test_data, eval_iters=50, n_feature=30, reg=5e-2, max_rating=5., min_rating=1., seed=0, n_iters=50):
    """Matrix factorization by ALS"""
    # Set seed
    rand_state = RandomState(0)

    (rows, cols, vals) = sp.find(data)
    ratings = np.array([np.array([int(u),int(i),int(r)]) for (u,i,r) in zip(cols, rows, vals)])

    (test_rows, test_cols, test_vals) = sp.find(test_data)
    test_ratings = np.array([np.array([int(u),int(i),int(r)]) for (u,i,r) in zip(test_cols, test_rows, test_vals)])

    n_user = max(max(ratings[:, 0]), max(test_ratings[:, 0])) + 1
    n_item = max(max(ratings[:, 1]), max(test_ratings[:, 1])) + 1

    als = ALS(n_user=n_user, n_item=n_item, n_feature=n_feature, reg=reg, max_rating=max_rating, min_rating=min_rating, seed=seed)

    als.fit(ratings, n_iters=n_iters)
    train_preds = als.predict(ratings[:, :2])
    train_rmse = RMSE(train_preds, ratings[:, 2])
    val_preds = als.predict(test_ratings[:, :2])
    val_rmse = RMSE(val_preds, test_ratings[:, 2])
    print("after %d iterations, train RMSE: %.6f, validation RMSE: %.6f" % \
      (eval_iters, train_rmse, val_rmse))

def recommend_ALS(data, test_data, n_feature=30, reg=5e-2, max_rating=5., min_rating=1., seed=0, n_iters=50,
    prediction_path=''):
    """Matrix factorization by ALS"""
    # Set seed
    rand_state = RandomState(0)

    (rows, cols, vals) = sp.find(data)
    ratings = np.array([np.array([int(u),int(i),int(r)]) for (u,i,r) in zip(cols, rows, vals)])

    (test_rows, test_cols, test_vals) = sp.find(test_data)
    test_ratings = np.array([np.array([int(u),int(i),int(r)]) for (u,i,r) in zip(test_cols, test_rows, test_vals)])

    n_user = max(ratings[:, 0]) + 1
    n_item = max(ratings[:, 1]) + 1

    als = ALS(n_user=n_user, n_item=n_item, n_feature=n_feature, reg=reg, max_rating=max_rating, min_rating=min_rating, seed=seed)

    als.fit(ratings, n_iters=n_iters)
    val_preds = als.predict(test_ratings[:, :2])

    with open('{dp}{fn}.csv'.format(dp=prediction_path, fn='newest'), 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for e in range(test_ratings.shape[0]):
            writer.writerow({'Id':'r{r}_c{c}'.format(r=test_ratings[e,1]+1,c=test_ratings[e,0]+1),'Prediction':val_preds[e]})
    # 0.98585 on Kaggle