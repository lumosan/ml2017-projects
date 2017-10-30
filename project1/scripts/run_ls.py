# run.py
import numpy as np
import matplotlib.pyplot as plt
#from custom_methods import auxiliary_methods, cost_gradient_methods, cross_validation_methods, data_processing_methods, proj1_helpers, standardizing_methods
from custom_methods.auxiliary_methods import *
from custom_methods.cost_gradient_methods import *
from custom_methods.cross_validation_methods import *
from custom_methods.data_processing_methods import *
from custom_methods.proj1_helpers import *
from custom_methods.standardizing_methods import *
from implementations import least_squares

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

def run_least_squares(x_1, test_x_1, Y, degree):
    """Auxiliary function to analyze least squares for a certain
    degree of polynomial basis.
    """
    ## We try Least Squares on tx building a polynomial basis
    poly_tx = build_poly(x_1, degree)
    test_poly_tx = build_poly(test_x_1, degree)

    # split into training, validation and testing
    np.random.seed(1)
    tv_tx, te_tx, tv_Y, te_Y = split_data(poly_tx, Y, .8)
    tr_tx, va_tx, tr_Y, va_Y = split_data(tv_tx, tv_Y, .8)

    # select labels with {-1,1} values
    tr_y = tr_Y.T[0]
    va_y = va_Y.T[0]
    te_y = te_Y.T[0]

    w, loss = least_squares(tr_y, tr_tx)

    return w, test_poly_tx


w, test_poly_tx = run_least_squares(x_1, test_x_1, Y, 8)

# predict labels and generate submission file
test_y_pred = predict_labels_bis(w, test_poly_tx)
create_csv_submission(test_ids, test_y_pred, DATA_PATH+'submission_ls.csv')
