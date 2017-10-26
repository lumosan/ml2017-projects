# Run script of the ML Project 01

import numpy as np

from proj1_helpers import *
from implementations import *
from auxiliary_functions import *
from visualization import *

DATA_PATH = '../data/'



def train_on_cross_validation():
	'''Function to handle the data via cross validation and return the best paramaters w'''

	# Cross validation
	seed = 13
	degree = 7
	k_fold = 10
	lambda_ = 10**(-2)

	ws = []

	# split data in k fold
	k_indices = build_k_indices(y, k_fold, seed)

	# K-fold cross validation and pick the rmse 'test' and 'train'
	# errors that represent the least absolute difference between
	# them (for a given lambda).
	for i, k in enumerate(range(k_fold)):
		w_opt, rmse_cur_tr, rmse_cur_te = cross_validation(y, x, k_indices, k, lambda_, degree)
		rmse_cur_dif = abs(rmse_cur_tr - rmse_cur_te)
		print('{}° w param result'.format(i+1))
		print('Difference: {}'.format(rmse_cur_dif), end='\n\n')

		ws.append(w_opt)

	# Avarage ?
	w_opt = sum(ws) / len(ws)
	print(w_opt)

	return w_opt

def train_on_cross_validation_with_many_lambdas():
	'''Function to handle the data via cross validation and return the best paramaters w'''

	# Cross validation
	seed = 13
	degree = 7
	k_fold = 10
	lambdas = np.logspace(-4, 2, 50)

	rmse_tr = []
	rmse_te = []

	# split data in k fold
	k_indices = build_k_indices(y, k_fold, seed)

	for lambda_ in lambdas:

		rmse_tr_for_this_lambda = []
		rmse_te_for_this_lambda = []

		# K-fold cross validation and pick the rmse 'test' and 'train'
		# errors that represent the least absolute difference between
		# them (for a given lambda).
		for i, k in enumerate(range(k_fold)):
			w_opt, rmse_cur_tr, rmse_cur_te = cross_validation(y, x, k_indices, k, lambda_, degree)

			rmse_tr_for_this_lambda.append(rmse_cur_tr)
			rmse_te_for_this_lambda.append(rmse_cur_te)

		# Avarage ?
		rmse_tr.append( sum(rmse_tr_for_this_lambda) / len(rmse_tr_for_this_lambda) )
		rmse_te.append( sum(rmse_te_for_this_lambda) / len(rmse_te_for_this_lambda) )

	
	print(rmse_tr)
	print(rmse_te)

	cross_validation_visualization(lambdas, rmse_tr, rmse_te)

	return w_opt




#####################
##### MAIN CODE #####
#####################

np.random.seed(1)

print('Running ...', end='\n\n')

# Import the data
yb, input_data, ids = load_csv_data('{}train.csv'.format(DATA_PATH), sub_sample=False)

test_yb, test_input_data, test_ids = load_csv_data( '{}test.csv'.format(DATA_PATH) )


# TODO: Once the Logistic regression is fixed we won't need this anymore
#y = np.ones(len(yb))
#y[np.where(yb==-1)] = 0

# Clean up the data (standardize, polynomial?, offset?, etc...)
x, mean_x, std_x = standardize(input_data)
y, tx = build_model_data(x, y)

test_x, test_mean_x, test_std_x = standardize(test_input_data)
test_y, test_tx = build_model_data(test_x, test_yb)

# Cross Validation
w_opt = train_on_cross_validation()

# Predict the labels
y_pred = predict_labels(w_opt, x_pred)


# Release a submission CSV with the predictions
create_csv_submission(ids, y_pred, '{}submission.csv'.format(DATA_PATH) )


print('Done :)')

