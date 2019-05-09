import numpy as np
from datafile_methods.data_io import prepare_data, load_datasets_sur
from prediction_methods.model_helpers import cross_validation, cross_validation_sur

from prediction_methods.baseline_model import model_baseline
from prediction_methods.als2_model import model_als2
from prediction_methods.svd1_model import model_svd1
from prediction_methods.surprise_models import model_knn
from prediction_methods.surprise_models import model_nmf
from prediction_methods.surprise_models import model_svd2


PREDICTION_PATH = 'data/predictions/'
DATA_PATH = 'data/'

# LOAD DATASETS
print('Loading datasets...')
folds, ratings, sample_submission = prepare_data(k=5, data_path=DATA_PATH)
folds_tr, folds_te, ratings_sur, sample_submission_sur = load_datasets_sur()


# OBTAIN MODEL PREDICTIONS

print('Baseline model started')
# 5-fold cross-validation
train_rmse, test_rmse = cross_validation(folds, model_baseline, {'prediction_path': PREDICTION_PATH})
# Train on entire training set. Predict for `submission` dataset
model_baseline(ratings, sample_submission, False, prediction_path=PREDICTION_PATH)


print('ALS2 model started')
u_feats = np.loadtxt("u_feats.txt")
i_feats = np.loadtxt("i_feats.txt")
args = {'prediction_path': PREDICTION_PATH,
        'k': 20,
        'lambda_u': .1,
        'lambda_i': .1,
        'tol': 1e-4,
        'max_iter': 100,
        'init_u_features': u_feats,
        'init_i_features': i_feats}
# 5-fold cross-validation
train_rmse, test_rmse = cross_validation(folds, model_als2, args)
# Train on entire training set. Predict for `submission` dataset
model_als2(ratings, sample_submission, False, **args)


print('SVD1 model started')
args = {'prediction_path': PREDICTION_PATH, 'k': 13}
# 5-fold cross-validation
train_rmse, test_rmse = cross_validation(folds, model_svd1, args)
# Train on entire training set. Predict for `submission` dataset
model_svd1(ratings, sample_submission, False, **args)


print('KNN_u model started')
args = {'prediction_path': PREDICTION_PATH,
        'k': 300,
        'min_k': 20,
        'name': 'pearson_baseline',
        'user_based': True,
        'fn_suffix': 'u_'}
# 5-fold cross-validation
train_rmse, test_rmse = cross_validation_sur(folds_tr, folds_te, model_knn, args)
# Train on entire training set. Predict for `submission` dataset
model_knn(ratings_sur, sample_submission_sur, False, **args)


print('KNN_i model started')
args = {'prediction_path': PREDICTION_PATH,
        'k': 60,
        'min_k': 20,
        'name': 'pearson_baseline',
        'user_based': False,
        'fn_suffix': 'i_'}
# 5-fold cross-validation
train_rmse, test_rmse = cross_validation_sur(folds_tr, folds_te, model_knn, args)
# Train on entire training set. Predict for `submission` dataset
model_knn(ratings_sur, sample_submission_sur, False, **args)


print('NMF model started')
args = {'prediction_path': PREDICTION_PATH,
        'biased': True,
        'k': 22,
        'reg_pu': 0.05,
        'reg_qi': 0.05,
        'reg_bu': 0.055,
        'reg_bi': 0.055,
        'n_epochs': 150}
# 5-fold cross-validation
train_rmse, test_rmse = cross_validation_sur(folds_tr, folds_te, model_nmf, args)
# Train on entire training set. Predict for `submission` dataset
model_nmf(ratings_sur, sample_submission_sur, False, **args)


print('SVD2 model started')
args = {'prediction_path': PREDICTION_PATH,
        'biased': True,
        'k': 130,
        'reg_all': 0.08,
        'n_epochs': 50}
# 5-fold cross-validation
train_rmse, test_rmse = cross_validation_sur(folds_tr, folds_te, model_svd2, args)
# Train on entire training set. Predict for `submission` dataset
model_svd2(ratings_sur, sample_submission_sur, False, **args)



print('Loading model ALS1...')
print('Loading model Slope One...')