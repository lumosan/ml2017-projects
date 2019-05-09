import numpy as np
from numpy.random import RandomState
from datafile_methods.data_io import save_csv_sur
from surprise import SlopeOne, KNNBaseline, NMF, SVD
from surprise.accuracy import rmse


def predict(algo, data, prediction_path, filename, save=True):
    # Do predictions for `data`
    pred = algo.test(data.build_testset())
    if save:
        # Write predictions to submission file
        save_csv_sur(pred, prediction_path=prediction_path, filename=filename)
    # Return pred
    pred = algo.test(data.build_testset())
    return pred


def model_slope_one(train_data, test_data, test_flag, prediction_path='',
    fold_number=''):
    """Uses Slope One algorithm from surprise library"""
    # Initialize algorithm
    algo_so = SlopeOne()
    # Train model
    algo_so.train(train_data)
    if test_flag:
        # Get train error
        train_pred = predict(algo_so, train_data, '', '', save=False)
        train_rmse = rmse(train_pred, verbose=False)
        # Get test error
        test_pred = predict(algo_so, test_data, prediction_path,
            'model_slope_one_te_{}'.format(fold_number))
        test_rmse = rmse(test_pred, verbose=False)
        return train_rmse, test_rmse
    else:
        # Create and save predictions as Kaggle submissions
        te_pred = predict(algo_so, test_data, prediction_path, 'model_slope_one_sub')


def model_knn(train_data, test_data, test_flag, prediction_path='',
    k=300, min_k=20, name='pearson_baseline',
    user_based=True, fn_suffix='', fold_number=''):
    """Uses KNN with baseline algorithm from surprise library"""
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
        train_pred = predict(algo_knn_bl, train_data, '', '', save=False)
        train_rmse = rmse(train_pred, verbose=False)
        # Get test error
        test_pred = predict(algo_knn_bl, test_data, prediction_path,
            'model_knn_{}te_{}'.format(fn_suffix, fold_number))
        test_rmse = rmse(test_pred, verbose=False)
        return train_rmse, test_rmse
    else:
        # Create and save predictions as Kaggle submissions
        te_pred = predict(algo_knn_bl, test_data, prediction_path,
            'model_knn_{}sub'.format(fn_suffix))


def model_nmf(train_data, test_data, test_flag, prediction_path='',
    biased=True, k=18, reg_pu=0.08, reg_qi=0.08,
    reg_bu=0.055, reg_bi=0.055, n_epochs=150, fold_number=''):
    """Uses NMF algorithm from surprise library"""
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
        train_pred = predict(algo_nmf, train_data, '', '', save=False)
        train_rmse = rmse(train_pred, verbose=False)
        # Get test error
        test_pred = predict(algo_nmf, test_data, prediction_path,
            'model_nmf_te_{}'.format(fold_number))
        test_rmse = rmse(test_pred, verbose=False)
        return train_rmse, test_rmse
    else:
        # Create and save predictions as Kaggle submissions
        te_pred = predict(algo_nmf, test_data, prediction_path, 'model_nmf_sub')


def model_svd2(train_data, test_data, test_flag, prediction_path='',
    biased=True, k=130, reg_all=0.065, n_epochs=50, fold_number=''):
    """Uses SVD algorithm from surprise library"""
    # Set seed and RandomState
    np.random.seed(0)
    rand_state = RandomState(0)
    # Initialize algorithm
    algo_svd = SVD(biased=biased, n_factors=k, reg_all=reg_all, n_epochs=n_epochs)
    # Train model
    algo_svd.train(train_data)
    if test_flag:
        # Get train error
        train_pred = predict(algo_svd, train_data, '', '', save=False)
        train_rmse = rmse(train_pred, verbose=False)
        # Get test error
        test_pred = predict(algo_svd, test_data, prediction_path,
            'model_svd2_te_{}'.format(fold_number))
        test_rmse = rmse(test_pred, verbose=False)
        return train_rmse, test_rmse
    else:
        # Create and save predictions as Kaggle submissions
        te_pred = predict(algo_svd, test_data, prediction_path, 'model_svd2_sub')
