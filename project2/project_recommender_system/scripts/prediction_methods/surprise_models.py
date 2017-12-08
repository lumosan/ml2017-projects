import numpy as np
from numpy.random import RandomState
from datafile_methods.data_io import save_csv_sur
from surprise import SlopeOne, CoClustering, KNNBaseline, NMF, SVD


def predict(algo, data, prediction_path, filename):
    # Do predictions for `data`
    pred = algo.test(data)
    # Write predictions to submission file
    save_csv_sur(pred, prediction_path=prediction_path, filename=filename)

def predict_error(algo, data):
    # Do predictions for `data`
    pred = algo.test(data)
    rmse = surprise.accuracy.rmse(pred, verbose=False)
    return rmse

def model_slope_one(train_data, test_data, test_flag, prediction_path=''):
    # Initialize algorithm
    algo_so = SlopeOne()
    # Train model
    algo_so.train(train_data)
    if test_flag:
        # Get train error
        train_rmse = predict_error(algo_so, train_data)
        # Get test error
        test_rmse = predict_error(algo_so, test_data)
        return train_rmse, test_rmse
    else:
        # Create and save predictions as Kaggle submissions
        predict(algo_so, test_data, prediction_path, 'model_slope_one_sub')
        predict(algo_so, train_data, prediction_path, 'model_slope_one_tr')

def model_co_clustering(train_data, test_data, test_flag, prediction_path='',
    n_cltr_u=75, n_cltr_i=3, n_epochs=100):
    # Set seed and RandomState
    np.random.seed(0)
    rand_state = RandomState(0)
    # Initialize algorithm
    algo_cc = CoClustering(n_cltr_u=75, n_cltr_i=3, n_epochs=100)
    # Train model
    algo_cc.train(train_data)
    if test_flag:
        # Get train error
        train_rmse = predict_error(algo_cc, train_data)
        # Get test error
        test_rmse = predict_error(algo_cc, test_data)
        return train_rmse, test_rmse
    else:
        # Create and save predictions as Kaggle submissions
        predict(algo_cc, test_data, prediction_path, 'model_co_clustering_sub')
        predict(algo_cc, train_data, prediction_path, 'model_co_clustering_tr')

def model_knn_baseline(train_data, test_data, test_flag, prediction_path='',
    k=300, min_k=20, name='pearson_baseline',
    user_based=True, fn_suffix=''):
    # Set seed and RandomState
    np.random.seed(0)
    rand_state = RandomState(0)
    # Initialize algorithm
    algo_knn_bl = KNNBaseline(k=k, min_k= min_k,
        sim_options={name:name, user_based:user_based})
    # Train model
    algo_knn_bl.train(train_data)
    if test_flag:
        # Get train error
        train_rmse = predict_error(algo_knn_bl, train_data)
        # Get test error
        test_rmse = predict_error(algo_knn_bl, test_data)
        return train_rmse, test_rmse
    else:
        # Create and save predictions as Kaggle submissions
        predict(algo_knn_bl, test_data, prediction_path,
            'model_knn_baseline_{}sub'.format(fn_suffix))
        predict(algo_knn_bl, train_data, prediction_path,
            'model_knn_baseline_{}tr'.format(fn_suffix))

def model_nmf(train_data, test_data, test_flag, prediction_path='',
    biased=True, k=18, reg_pu=0.08, reg_qi=0.08,
    reg_bu=0.055, reg_bi=0.055, n_epochs=150):
    """NMF collaborative filtering"""
    # Set seed and RandomState
    np.random.seed(0)
    rand_state = RandomState(0)
    # Initialize algorithm
    algo_nmf = NMF(biased=biased, n_factors=k, reg_pu=reg_pu, reg_qi=reg_qi,
        reg_bu=reg_bu, reg_bi=reg_bi, n_epochs=n_epochs)
    # Train model
    algo_nmf.train(train_data)
    if test_flag:
        # Get train error
        train_rmse = predict_error(algo_nmf, train_data)
        # Get test error
        test_rmse = predict_error(algo_nmf, test_data)
        return train_rmse, test_rmse
    else:
        # Create and save predictions as Kaggle submissions
        predict(algo_nmf, test_data, prediction_path, 'model_nmf_sub')
        predict(algo_nmf, train_data, prediction_path, 'model_nmf_tr')

def model_svd(train_data, test_data, test_flag, prediction_path='',
    biased=True, k=130, reg_all=0.065, n_epochs=50):
    """NMF collaborative filtering"""
    # Set seed and RandomState
    np.random.seed(0)
    rand_state = RandomState(0)
    # Initialize algorithm
    algo_svd = SVD(biased=biased, n_factors=k, reg_all=reg_all, n_epochs=n_epochs)
    # Train model
    algo_svd.train(train_data)
    if test_flag:
        # Get train error
        train_rmse = predict_error(algo_svd, train_data)
        # Get test error
        test_rmse = predict_error(algo_svd, test_data)
        return train_rmse, test_rmse
    else:
        # Create and save predictions as Kaggle submissions
        predict(algo_svd, test_data, prediction_path, 'model_sur_svd_sub')
        predict(algo_svd, train_data, prediction_path, 'model_sur_svd_tr')
