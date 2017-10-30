# Analysis of least_squares
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from auxiliary_functions import *
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


def try_least_squares(x_1, Y, degree):
    """Auxiliary function to analyze least squares for a certain
    degree of polynomial basis.
    """
    ## We try Least Squares on tx building a polynomial basis
    poly_tx = build_poly(x_1, degree)

    # split into training, validation and testing
    np.random.seed(1)
    tv_tx, te_tx, tv_Y, te_Y = split_data(poly_tx, Y, .8)
    tr_tx, va_tx, tr_Y, va_Y = split_data(tv_tx, tv_Y, .8)

    # select labels with {-1,1} values
    tr_y = tr_Y.T[0]
    va_y = va_Y.T[0]
    te_y = te_Y.T[0]

    w, loss = least_squares(tr_y, tr_tx)

    # calculate and print precisions obtained
    tr_pr = calculate_precision(predict_labels(w, tr_tx), tr_y)
    va_pr = calculate_precision(predict_labels(w, va_tx), va_y)
    te_pr = calculate_precision(predict_labels(w, te_tx), te_y)

    print("Training={tr}, Validation={va}".format(tr=tr_pr, va=va_pr))
    return (tr_pr, va_pr, te_pr)


def analyze_least_squares(x_1, Y):
    """Auxiliary function to analyze least squares for different
    degrees of polynomial basis.
    """
    tr_prs_ls = []
    va_prs_ls = []
    te_prs_ls = []
    for d in range(1,9):
        tr_pr, va_pr, te_pr = try_least_squares(x_1, Y, d)
        tr_prs_ls.append(tr_pr)
        va_prs_ls.append(va_pr)
        te_prs_ls.append(te_pr)

    # plot precision for tr, va, and te
    plt.plot(range(1,9), tr_prs_ls, label="Train")
    plt.plot(range(1,9), va_prs_ls, label="Validation")
    plt.plot(range(1,9), te_prs_ls, label="Test")
    plt.title("Precision for Least Squares")
    plt.xlabel('Maximum degree')
    plt.yticks(np.arange(0.74, 0.8, .01))
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('LSprec.pdf')


## We try Least Squares on tx for polynomial basis of different degrees
analyze_least_squares(x_1, Y)
