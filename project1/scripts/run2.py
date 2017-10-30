# run.py
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from auxiliary_functions import *
from implementations import least_squares
from implementations import reg_logistic_regression_improved

## Read data
DATA_PATH = '../data/'
# read files and prepare data
yb, input_data, ids = load_csv_data(DATA_PATH+'train.csv')
test_yb, test_input_data, test_ids = load_csv_data(DATA_PATH+'test.csv')

# standardize and remove nans
x_1, mean_x_1, std_x_1 = standardize_ignoring_values(input_data, -999)
test_x_1 = standardize_test_ignoring_values(test_input_data, -999, mean_x_1, std_x_1)

# Y contains two arrays, one with labels -1 and 1, other with 0 and 1
y = np.ones(len(yb))
y[np.where(yb==-1)] = 0
Y=np.array((yb, y)).T

def run_logistic_regression(x_1, test_x_1, Y, degree, max_iters, gamma):
    """Auxiliary function to analyze logistic regression for a certain
    degree of polynomial basis.
    """
    # build polynomial basis
    poly_tx, test_poly_tx = build_poly_PCA(x_1, test_x_1, degree, 0,
        apply_pca=False)

    # split into training, validation and testing
    np.random.seed(1)
    tv_tx, te_tx, tv_Y, te_Y = split_data(poly_tx, Y, .8)
    tr_tx, va_tx, tr_Y, va_Y = split_data(tv_tx, tv_Y, .8)

    # select labels with {0,1} values
    tr_y = tr_Y.T[1]
    va_y = va_Y.T[1]
    te_y = te_Y.T[1]

    # obtain model with the chosen hyperparameters
    np.random.seed(1)
    lambda_= 0
    initial_w = np.zeros((tr_tx.shape[1], 1))
    w, loss = reg_logistic_regression_improved(tr_y, tr_tx, lambda_,
        initial_w, max_iters, gamma, batch_size=2000)

    return w, test_poly_tx


w, test_poly_tx = run_logistic_regression(x_1, test_x_1, Y, 5, 1500,
        adaptive_gamma(kappa=0.8, eta0=1e-2))

# predict labels and generate submission file
test_y_pred = predict_labels_bis(w, test_poly_tx)
create_csv_submission(test_ids, test_y_pred, DATA_PATH+'submission.csv')
