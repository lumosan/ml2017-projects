import numpy as np
import scipy.sparse as sp
from itertools import groupby
from itertools import compress
import pandas as pd

from sklearn.metrics import mean_squared_error
from math import sqrt

from datafile_methods.data_io import save_csv
from functools import reduce

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


def disjoint_subsets_on_support(ratings, indices, return_vals=True,
    cols=None, rows=None):
    """Returns two lists of values, the first has data for columns
    included in indices (high support ratings), the second the remaining
    (low support ratings).
    """
    # Split the data into two subsets
    # initialize sparse matrix
    ratings_high = sp.lil_matrix(ratings.shape)
    # copy relevant items to sparse matrices
    ratings_high[:,indices] = ratings[:,indices]
    ratings_low = ratings - ratings_high

    if return_vals:
        if (rows is None) or (cols is None):
            # Get lists of values, assumes non-zero relevant elements
            ratings_high_vals = sp.find(ratings_high)[2]
            ratings_low_vals = sp.find(ratings_low)[2]
            return ratings_high_vals, ratings_low_vals
        else:
            ind_h = [u in indices for u in cols]
            rows_h = list(compress(rows, ind_h))
            cols_h = list(compress(cols, ind_h))
            ratings_high_vals = ratings_high[rows_h, cols_h]
            ind_l = np.invert(ind_h)
            rows_l = list(compress(rows, ind_l))
            cols_l = list(compress(cols, ind_l))
            ratings_low_vals = ratings_low[rows_l, cols_l]
            return ratings_high_vals, ratings_low_vals
    else:
        # Returns entire matrices
        return ratings_high, ratings_low

def disjoint_subsets_on_support_sub(ratings, indices):
    """Returns two lists of values, the first has data for columns
    included in indices (high support ratings), the second the remaining
    (low support ratings).
    """
    # Split the data into two subsets
    # initialize sparse matrix
    ratings_high = sp.lil_matrix(ratings.shape)
    # copy relevant items to sparse matrices
    ratings_high[:,indices] = ratings[:,indices]
    ratings_low = ratings - ratings_high

    return ratings_high, ratings_low

def obtain_indices_high_support(all_ratings, val=1802):
    """Returns a list of the users with the highest support.
    Suport of a rating (i,u) is the number of ratings user u has given.
    """
    # Obtain number of ratings per user
    num_ratings_per_user = np.array((all_ratings != 0).sum(axis=0)).flatten()
    # Get users with highest support
    high_support_bool = num_ratings_per_user > val
    high_support_users = list(compress(range(len(high_support_bool)), high_support_bool))
    indices = high_support_users
    return indices


def create_weighted_ensemble_submission(ratings, predictions_sub_dict,
    high_w=None, low_w=None, prediction_path=''):
    """Creates an ensemble by computing the weighted mean
    of the predictions in `predictions_sub_dict`.
    """
    models = list(predictions_sub_dict.keys())

    # Obtain list of indices for high support ratings
    ind = obtain_indices_high_support(ratings)

    # Separate submission data into high and low support ratings
    predictions_sub = {k: disjoint_subsets_on_support(v, ind,
        return_vals=False) for k,v in predictions_sub_dict.items()}
    predictions_sub_high = {k: v[0] for k,v in predictions_sub.items()}
    predictions_sub_low = {k: v[1] for k,v in predictions_sub.items()}

    if high_w is None:
        # Define weights as estimated
        high_w = {'baseline': 0,
                  'knn_baseline_u': 0,
                  'sur_svd': 1.0/2,
                  'mf_als_recommend': 1.0/8,
                  'mf_als': 1.0/8,
                  'knn_baseline_i': 1.0/16,
                  'mf_svd_sci': 1.0/16,
                  'nmf': 1.0/16,
                  'slope_one': 1.0/16}

    if low_w is None:
        # Define weights as estimated
        low_w = {'mf_als': 1.0/3,
                 'baseline': 0,
                 'knn_baseline_u': 0,
                 'nmf': 0,
                 'slope_one': 0,
                 'knn_baseline_i': 0,
                 'sur_svd': 1.0/3,
                 'mf_als_recommend': 1.0/6,
                 'mf_svd_sci': 1.0/6}

    # Obtain weighted mean predictions for submission file
    sp_mean_sub_high = np.sum([(high_w[m] * predictions_sub_high[m]) for m in models])
    sp_mean_sub_low = np.sum([(low_w[m] * predictions_sub_low[m]) for m in models])
    # Join predictions for high and low support
    sp_mean_sub = sp_mean_sub_high + sp_mean_sub_low

    # Export file
    save_csv(sp_mean_sub, prediction_path=prediction_path, filename='combination_manual')

def evaluate_manual_weighted_ensemble(ratings, folds, predictions_dict):
    def compute_mean_predictions(predictions_df, chosen_models):
        return predictions_df[chosen_models].mean(axis=1)

    def compute_mean_fold_predictions(predictions_df_list, chosen_models):
        return [compute_mean_predictions(f, chosen_models)
                for f in predictions_df_list]

    def evaluate_partial_ensembles_high():
        predictions_high_df = preds_high_df.copy()
        c_models_h = [['knn_baseline_i', 'mf_svd_sci'],
                     ['nmf', 'slope_one'],
                     ['mf_als_recommend', 'comb1', 'comb2', 'mf_als'],
                     ['comb3', 'sur_svd']]

        comb1 = compute_mean_fold_predictions(predictions_high_df, c_models_h[0])
        # Get test error
        error_comb1 = [sqrt(mean_squared_error(observations_high_df[i], comb1[i]))
            for i in range(k_fold)]

        comb2 = compute_mean_fold_predictions(predictions_high_df, c_models_h[1])
        # Get test error
        error_comb2 = [sqrt(mean_squared_error(observations_high_df[i], comb2[i]))
            for i in range(k_fold)]

        # Save combinations of predictions in the predictions DataFrame
        for i in range(k_fold):
            predictions_high_df[i]['comb1'] = pd.Series(comb1[i])
            predictions_high_df[i]['comb2'] = pd.Series(comb2[i])

        comb3 = compute_mean_fold_predictions(predictions_high_df, c_models_h[2])
        # Get test error
        error_comb3 = [sqrt(mean_squared_error(observations_high_df[i], comb3[i]))
            for i in range(k_fold)]

        # Save combination of predictions in the predictions DataFrame
        for i in range(5):
            predictions_high_df[i]['comb3'] = pd.Series(comb3[i])

        comb4 = compute_mean_fold_predictions(predictions_high_df, c_models_h[3])
        # Get test error
        error_comb4 = [sqrt(mean_squared_error(observations_high_df[i], comb4[i]))
            for i in range(k_fold)]

        return predictions_high_df, [error_comb1, error_comb2, error_comb3, error_comb4]

    def evaluate_partial_ensembles_low():
        predictions_low_df = preds_low_df.copy()
        c_models_l = [['mf_als_recommend', 'mf_svd_sci'],
                      ['mf_als', 'sur_svd', 'comb1']]

        comb1 = compute_mean_fold_predictions(predictions_low_df, c_models_l[0])
        # Get test error
        error_comb1 = [sqrt(mean_squared_error(observations_low_df[i], comb1[i]))
            for i in range(k_fold)]

        # Save combination of predictions in the predictions DataFrame
        for i in range(k_fold):
            predictions_low_df[i]['comb1'] = pd.Series(comb1[i])

        comb2 = compute_mean_fold_predictions(predictions_low_df, c_models_l[1])

        error_comb2 = [sqrt(mean_squared_error(observations_low_df[i], comb2[i]))
            for i in range(k_fold)]

        return predictions_low_df, [error_comb1, error_comb2]


    k_fold = len(folds)
    models = list(predictions_dict.keys())

    # Obtain list of indices for high support ratings
    ind = obtain_indices_high_support(ratings)

    # Split observed ratings according to support level
    observations_high = [disjoint_subsets_on_support(f, ind)[0] for f in folds]
    observations_low = [disjoint_subsets_on_support(f, ind)[1] for f in folds]

    # Split predictions according to support level
    predictions_high = {k: [disjoint_subsets_on_support(v[i], ind)[0]
        for i in range(k_fold)] for k,v in predictions_dict.items()}
    predictions_low = {k: [disjoint_subsets_on_support(v[i], ind)[1]
        for i in range(k_fold)] for k,v in predictions_dict.items()}

    # Create dataframes with observed values
    observations_high_df = [pd.DataFrame(f) for f in observations_high]
    observations_low_df = [pd.DataFrame(f) for f in observations_low]

    # Create dataframes with predicted values
    preds_high_df = [pd.DataFrame({model: obs[i] for model, obs in
        predictions_high.items()}) for i in range(k_fold)]
    preds_low_df = [pd.DataFrame({model: obs[i] for model, obs in
        predictions_low.items()}) for i in range(k_fold)]

    # Get errors and prediction values for each partial and final ensemble
    predictions_high_df, errors_comb_high = evaluate_partial_ensembles_high()
    predictions_low_df, errors_comb_low = evaluate_partial_ensembles_low()

    return predictions_high_df, errors_comb_high, predictions_low_df, errors_comb_low



def compute_meta_features(train_data, test_data):
    """Compute some meta features based on the values of `train_data`
    for `test_data`. Both of the inputs must be sparse matrices.
    It returns arrays with the meta-features values, in the order returned
    by sp.find()
    """
    n_i_tr = np.count_nonzero(train_data.toarray(), axis=1)
    n_u_tr = np.count_nonzero(train_data.toarray(), axis=0)

    # When getting the std, if there are no ratings we set value 0
    std_i = np.array([np.std(sp.find(i)[2]) if i.count_nonzero() > 0
        else 0 for i in train_data])
    std_u = np.array([np.std(sp.find(u)[2]) if u.count_nonzero() > 0
        else 0 for u in train_data.T])
    log_i = np.array([np.log(i.count_nonzero()) if i.count_nonzero() > 0
        else 0 for i in train_data])
    log_u = np.array([np.log(u.count_nonzero()) if u.count_nonzero() > 0
        else 0 for u in train_data.T])

    # Get non-zero elements
    (test_rows, test_cols, test_vals) = sp.find(test_data)

    # Obtain meta features for `test_data`
    n_i_val = [n_i_tr[i] for (i, u) in zip(test_rows, test_cols)]
    n_u_val = [n_u_tr[u] for (i, u) in zip(test_rows, test_cols)]
    std_i_val = np.array([std_i[i] for (i, u) in zip(test_rows, test_cols)])
    std_u_val = np.array([std_u[u] for (i, u) in zip(test_rows, test_cols)])
    log_i_val = np.array([log_i[i] for (i, u) in zip(test_rows, test_cols)])
    log_u_val = np.array([log_u[u] for (i, u) in zip(test_rows, test_cols)])

    n_i = sp.csr_matrix((n_i_val, (test_rows, test_cols)), shape=test_data.shape)
    n_u = sp.csr_matrix((n_u_val, (test_rows, test_cols)), shape=test_data.shape)
    std_i = sp.csr_matrix((std_i_val, (test_rows, test_cols)), shape=test_data.shape)
    std_u = sp.csr_matrix((std_u_val, (test_rows, test_cols)), shape=test_data.shape)
    log_i = sp.csr_matrix((log_i_val, (test_rows, test_cols)), shape=test_data.shape)
    log_u = sp.csr_matrix((log_u_val, (test_rows, test_cols)), shape=test_data.shape)

    result = {'n_i': n_i, 'n_u': n_u, 'std_i': std_i,
        'std_u': std_u, 'log_i': log_i, 'log_u': log_u}

    return result, test_rows, test_cols

def compute_meta_features_per_fold(folds):
    """Gets meta-features for test set by using train set,
    for all k folds.
    Returns a lists of k dataframes.
    """
    def eval_meta_feat_fold_i(i):
        folds_copy = folds.copy()
        test = folds_copy.pop(i)
        train = reduce(lambda x,y: x+y, [m for m in folds_copy])
        [res, rs, cs] = compute_meta_features(train, test)
        return [res, rs, cs]

    meta_feats = [eval_meta_feat_fold_i(i)[0] for i in range(len(folds))]
    rows = [eval_meta_feat_fold_i(i)[1] for i in range(len(folds))]
    cols = [eval_meta_feat_fold_i(i)[2] for i in range(len(folds))]
    return [meta_feats, rows, cols]

def evaluate_meta_features_ensemble(ratings, folds, predictions_dict):
    k_fold = len(folds)
    models = list(predictions_dict.keys())

    # Obtain list of indices for high support ratings
    ind = obtain_indices_high_support(ratings)
    # Split observed ratings according to support level
    observations_high_sp = [disjoint_subsets_on_support(f, ind,
        return_vals=False)[0] for f in folds]
    observations_low_sp = [disjoint_subsets_on_support(f, ind,
        return_vals=False)[1] for f in folds]

    # Split predictions according to support level
    predictions_high = {k: [disjoint_subsets_on_support(v[i], ind)[0]
        for i in range(k_fold)] for k,v in predictions_dict.items()}
    predictions_low = {k: [disjoint_subsets_on_support(v[i], ind)[1]
        for i in range(k_fold)] for k,v in predictions_dict.items()}

    # Compute meta-features
    meta_features_sp, rows, cols = compute_meta_features_per_fold(folds)
    # Split meta-features according to support level
    meta_features_all = [{k: disjoint_subsets_on_support(v, ind) for k,v in meta_features_sp[f].items()} for f in range(k_fold)]
    meta_features_high = [{k: v[0] for k,v in meta_features_all[f].items()} for f in range(k_fold)]
    meta_features_low = [{k: v[1] for k,v in meta_features_all[f].items()} for f in range(k_fold)]
    # Get DataFrames with meta-features
    meta_features_high_df = [pd.DataFrame(f) for f in meta_features_high]
    meta_features_low_df = [pd.DataFrame(f) for f in meta_features_low]

    # Get observed values
    observations_high = [sp.find(e)[2] for e in observations_high_sp]
    observations_low = [sp.find(e)[2] for e in observations_low_sp]
    # Create dataframes with observed values
    observations_high_df = [pd.DataFrame(f) for f in observations_high]
    observations_low_df = [pd.DataFrame(f) for f in observations_low]

    # Create dataframes with predicted values
    preds_high_df = [pd.DataFrame({model: obs[i] for model, obs in
        predictions_high.items()}) for i in range(k_fold)]
    preds_low_df = [pd.DataFrame({model: obs[i] for model, obs in
        predictions_low.items()}) for i in range(k_fold)]

    # Concatenate predictors (meta_features and model predictions)
    predictors_high_df = [pd.concat([meta_features_high_df[i], preds_high_df[i]], axis=1)
        for i in range(k_fold)]
    predictors_low_df = [pd.concat([meta_features_low_df[i], preds_low_df[i]], axis=1)
        for i in range(k_fold)]
    return predictors_high_df, observations_high_df, predictors_low_df, observations_low_df


def create_sklearn_ensemble_submission(ratings, predictions_sub_dict,
    predictions_dict, prediction_path=''):
    """Creates an ensemble by computing the weighted mean
    of the predictions in `predictions_sub_dict`.
    """
    def get_features_train():
        # Obtain features and observed ratings for training data
        # Separate training data into high and low support ratings
        predictions_dict_unified = {k: reduce(lambda x,y: x+y, [m for m in v])
            for k,v in predictions_dict.items()}
        predictions_tr = {k: disjoint_subsets_on_support_sub(v, ind)
            for k,v in predictions_dict_unified.items()}
        predictions_tr_high = {k: sp.find(v[0])[2] for k,v in predictions_tr.items()}
        predictions_tr_low = {k: sp.find(v[1])[2] for k,v in predictions_tr.items()}
        # Create dataframes with predicted values
        predictions_tr_high_df = pd.DataFrame(predictions_tr_high)
        predictions_tr_low_df = pd.DataFrame(predictions_tr_low)

        # Get rows and columns of elements for each support level
        rows_cols = [sp.find(m)[0:2] for m in predictions_tr[models[0]]]

        # Compute meta-features
        meta_features_tr_sp, r, c = compute_meta_features(ratings, ratings)
        # Split meta-features according to support level
        meta_features_tr_all = {k: disjoint_subsets_on_support_sub(v, ind)
            for k,v in meta_features_tr_sp.items()}
        meta_features_tr_high = {k: sp.find(v[0])[2] for k,v in
            meta_features_tr_all.items()}
        meta_features_tr_low = {k: sp.find(v[1])[2] for k,v in
            meta_features_tr_all.items()}
        # Get DataFrames with meta-features
        meta_features_tr_high_df = pd.DataFrame(meta_features_tr_high)
        meta_features_tr_low_df = pd.DataFrame(meta_features_tr_low)

        # Concatenate features (meta_features and model predictions)
        features_tr_high_df = pd.concat([meta_features_tr_high_df,
            predictions_tr_high_df], axis=1)
        features_tr_low_df = pd.concat([meta_features_tr_low_df,
            predictions_tr_low_df], axis=1)
        return features_tr_high_df, features_tr_low_df, rows_cols

    def get_features_sub():
        # Obtain features for submission data
        # Separate submission data into high and low support ratings
        predictions_sub = {k: disjoint_subsets_on_support_sub(v, ind)
            for k,v in predictions_sub_dict.items()}
        predictions_sub_high = {k: sp.find(v[0])[2] for k,v in predictions_sub.items()}
        predictions_sub_low = {k: sp.find(v[1])[2] for k,v in predictions_sub.items()}
        # Create dataframes with predicted values
        predictions_sub_high_df = pd.DataFrame(predictions_sub_high)
        predictions_sub_low_df = pd.DataFrame(predictions_sub_low)

        # Get rows and columns of elements for each support level
        rows_cols = [sp.find(m)[0:2] for m in predictions_sub[models[0]]]

        # Compute meta-features
        meta_features_sub_sp, r, c = compute_meta_features(ratings,
            predictions_sub_dict[models[0]])
        # Split meta-features according to support level
        meta_features_sub_all = {k: disjoint_subsets_on_support(v, ind)
            for k,v in meta_features_sub_sp.items()}
        meta_features_sub_high = {k: v[0] for k,v in meta_features_sub_all.items()}
        meta_features_sub_low = {k: v[1] for k,v in meta_features_sub_all.items()}
        # Get DataFrames with meta-features
        meta_features_sub_high_df = pd.DataFrame(meta_features_sub_high)
        meta_features_sub_low_df = pd.DataFrame(meta_features_sub_low)

        # Concatenate features (meta_features and model predictions)
        features_sub_high_df = pd.concat([meta_features_sub_high_df,
            predictions_sub_high_df], axis=1)
        features_sub_low_df = pd.concat([meta_features_sub_low_df,
            predictions_sub_low_df], axis=1)
        return features_sub_high_df, features_sub_low_df, rows_cols

    def fit_and_predict(feat_tr, vals_tr, feat_sub, args={}):
        # Create and fit model
        clf = Ridge(**args)
        train = poly.fit_transform(feat_tr)
        test = poly.fit_transform(feat_sub)
        clf.fit(train, vals_tr)
        # Obtain predictions for submission data
        pred_sub = clf.predict(test)
        pred_sub = np.clip(pred_sub, 1.0, 5.0)
        return pred_sub

    models = list(predictions_sub_dict.keys())
    poly = PolynomialFeatures(2)

    # Obtain list of indices for high support ratings
    ind = obtain_indices_high_support(ratings)
    feat_high_tr, feat_low_tr, rows_cols_tr = get_features_train()
    feat_high_sub, feat_low_sub, rows_cols_sub = get_features_sub()

    # Get observed values for high and low support ratings
    vals_high_tr = np.array(ratings[rows_cols_tr[0]]).flatten()
    vals_low_tr = np.array(ratings[rows_cols_tr[1]]).flatten()

    # Obtain predictions for submission data
    args_h = {'alpha': 0.000125, 'normalize': True}
    pred_high_sub = fit_and_predict(feat_high_tr, vals_high_tr,
        feat_high_sub, args=args_h)
    args_l = {'alpha': 0.000375, 'normalize': True}
    pred_low_sub = fit_and_predict(feat_low_tr, vals_low_tr,
        feat_low_sub, args=args_l)

    # Get predictions into sparse matrix form
    pred_high_sp = sp.csr_matrix((pred_high_sub, rows_cols_sub[0]),
        shape=predictions_sub_dict[models[0]].shape)
    pred_low_sp = sp.csr_matrix((pred_low_sub, rows_cols_sub[1]),
        shape=predictions_sub_dict[models[0]].shape)

    # Join predictions for high and low support
    pred_sp = pred_high_sp + pred_low_sp

    # Export file
    save_csv(pred_sp, prediction_path=prediction_path, filename='combination_ridge')
