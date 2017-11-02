# run.py
import numpy as np
import matplotlib.pyplot as plt
from custom_methods.auxiliary_methods import *
from custom_methods.cost_gradient_methods import *
from custom_methods.cross_validation_methods import *
from custom_methods.data_processing_methods import *
from custom_methods.proj1_helpers import *
from custom_methods.standardizing_methods import *
from implementations import reg_logistic_regression_improved

## Read data
DATA_PATH = '../data/'
# read files and prepare data
yb, input_data, ids = load_csv_data(DATA_PATH+'train.csv')
test_yb, test_input_data, test_ids = load_csv_data(DATA_PATH+'test.csv')


def run_logistic_regression(x_1, test_x_1, y, degree, max_iters, gamma):
    """Auxiliary function to analyze logistic regression for a certain
    degree of polynomial basis.
    """
    # build polynomial basis
    poly_tx, test_poly_tx = build_poly_PCA(x_1, test_x_1, degree, 0,
        apply_pca=False)

    # split into training, validation and testing
    np.random.seed(1)
    tv_tx, te_tx, tv_y, te_y = split_data(poly_tx, y, .8)
    tr_tx, va_tx, tr_y, va_y = split_data(tv_tx, tv_y, .8)

    # obtain model with the chosen hyperparameters
    np.random.seed(1)
    lambda_= 0
    initial_w = np.zeros((tr_tx.shape[1], 1))
    w, loss = reg_logistic_regression_improved(tr_y, tr_tx, lambda_,
        initial_w, max_iters, gamma, batch_size=2000)

    tr_ac = calculate_precision(tr_y, predict_labels_bis(w, tr_tx, return_zeros=True), zeros_ones=True)
    va_ac = calculate_precision(va_y, predict_labels_bis(w, va_tx, return_zeros=True), zeros_ones=True)
    te_ac = calculate_precision(te_y, predict_labels_bis(w, te_tx, return_zeros=True), zeros_ones=True)
    print("TR:{tr}  VA:{va}  TE:{te}".format(tr=tr_ac, va=va_ac, te=te_ac))

    return w, test_poly_tx


empty_columns = [[4, 5, 6, 12, 22, 26, 27, 28], [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29, 30], []]

# divide data according to missing values
input_data_s = [input_data[np.where((input_data.T[4]==-999) & (input_data.T[23]!=-999))],
    input_data[np.where(input_data.T[23]==-999)],
    input_data[np.where((input_data.T[4]!=-999) & (input_data.T[23]!=-999))]]

# transform y to labels 0 and 1
y = np.ones(len(yb))
y[np.where(yb==-1)] = 0

y_s = [y[np.where((input_data.T[4]==-999) & (input_data.T[23]!=-999))],
    y[np.where(input_data.T[23]==-999)],
    y[np.where((input_data.T[4]!=-999) & (input_data.T[23]!=-999))]]

# divide test data according to missing values
test_input_data_s = [test_input_data[np.where((test_input_data.T[4]==-999) & (test_input_data.T[23]!=-999))],
    test_input_data[np.where(test_input_data.T[23]==-999)],
    test_input_data[np.where((test_input_data.T[4]!=-999) & (test_input_data.T[23]!=-999))]]

test_ids_s = [test_ids[np.where((test_input_data.T[4]==-999) & (test_input_data.T[23]!=-999))],
    test_ids[np.where(test_input_data.T[23]==-999)],
    test_ids[np.where((test_input_data.T[4]!=-999) & (test_input_data.T[23]!=-999))]]

# initialize arrays
w_s = []
test_poly_tx_s = []
test_y_pred_s = []

# define the adaptive gammas
gammas = [adaptive_gamma(kappa=0.4, eta0=1e-2), adaptive_gamma(kappa=0.3, eta0=1e-2), adaptive_gamma(kappa=0.6, eta0=1e-2)]

# run logistic regression for each subgroup of data
for i in range(3):
    # select relevant columns
    x_matrix = input_data_s[i][:,[x for x in range(30) if x not in empty_columns[i]]]
    test_x_matrix = test_input_data_s[i][:,[x for x in range(30) if x not in empty_columns[i]]]
    # standardize
    x_1, mean, std = standardize_ignoring_values(x_matrix, -999)
    test_x_1 = standardize_test_ignoring_values(test_x_matrix, -999, mean, std)
    # calculate weights
    w, test_poly_tx = run_logistic_regression(x_1, test_x_1, y_s[i], 5, 1500,
        gammas[i])
    w_s.append(w)
    test_poly_tx_s.append(test_poly_tx)
    test_y_pred = predict_labels_bis(w, test_poly_tx)
    test_y_pred_s.append(test_y_pred)

ti = np.array([])
tp = np.array([])
for i in range(3):
    ti = np.concatenate((ti, test_ids_s[i]))
    tp = np.concatenate((tp, np.squeeze(test_y_pred_s[i])))

# predict labels and generate submission file
create_csv_submission(ti, tp, DATA_PATH+'submission.csv')