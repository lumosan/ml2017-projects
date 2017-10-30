# run.py
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from auxiliary_functions import *
from implementations import reg_logistic_regression_improved

## Read data
DATA_PATH = '../data/'
# read files and prepare data
yb, input_data, ids = load_csv_data(DATA_PATH+'train.csv')
test_yb, test_input_data, test_ids = load_csv_data(DATA_PATH+'test.csv')

# standardize and remove nans
x_1, mean_x_1, std_x_1 = standardize_ignoring_values(input_data, -999)
test_x_1 = standardize_test_ignoring_values(test_input_data,
    -999, mean_x_1, std_x_1)

# Y contains two arrays, one with labels -1 and 1, other with 0 and 1
y = np.ones(len(yb))
y[np.where(yb==-1)] = 0
Y=np.array((yb, y)).T


def try_logistic_regression(x_1, test_x_1, Y, degree, max_iters, gamma):
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

    # calculate and print precisions obtained
    tr_pr = calculate_precision(predict_labels_bis(w, tr_tx,
        return_zeros=True), tr_y, zeros_ones=True)
    va_pr = calculate_precision(predict_labels_bis(w, va_tx,
        return_zeros=True), va_y, zeros_ones=True)
    te_pr = calculate_precision(predict_labels_bis(w, te_tx,
        return_zeros=True), te_y, zeros_ones=True)

    print("Training={tr}, Validation={va}, Test={te}".format(tr=tr_pr, va=va_pr, te=te_pr))
    return (tr_pr, va_pr, te_pr)


def analyze_logistic_regression(x_1, Y):
    """Auxiliary function to analyze logistic regression for different
    degree of polynomial basis.
    """
    tr_prs_lr = []
    va_prs_lr = []
    te_prs_lr = []

    (tr_pr, va_pr, te_pr) = try_logistic_regression(x_1, test_x_1, Y, 1, 1000,
        adaptive_gamma(kappa=0.8, eta0=1e-3))
    tr_prs_lr.append(tr_pr)
    va_prs_lr.append(va_pr)
    te_prs_lr.append(te_pr)

    (tr_pr, va_pr, te_pr) = try_logistic_regression(x_1, test_x_1, Y, 2, 1000,
        adaptive_gamma(kappa=0.8, eta0=1e-2))
    tr_prs_lr.append(tr_pr)
    va_prs_lr.append(va_pr)
    te_prs_lr.append(te_pr)

    (tr_pr, va_pr, te_pr) = try_logistic_regression(x_1, test_x_1, Y, 3, 1000,
        adaptive_gamma(kappa=0.8, eta0=1e-2))
    tr_prs_lr.append(tr_pr)
    va_prs_lr.append(va_pr)
    te_prs_lr.append(te_pr)

    (tr_pr, va_pr, te_pr) = try_logistic_regression(x_1, test_x_1, Y, 4, 1000,
        adaptive_gamma(kappa=0.8, eta0=1e-2))
    tr_prs_lr.append(tr_pr)
    va_prs_lr.append(va_pr)
    te_prs_lr.append(te_pr)

    (tr_pr, va_pr, te_pr) = try_logistic_regression(x_1, test_x_1, Y, 5, 1500,
        adaptive_gamma(kappa=0.8, eta0=1e-2))
    tr_prs_lr.append(tr_pr)
    va_prs_lr.append(va_pr)
    te_prs_lr.append(te_pr)

    (tr_pr, va_pr, te_pr) = try_logistic_regression(x_1, test_x_1, Y, 6, 1500,
        adaptive_gamma(kappa=0.8, eta0=1e-2))
    tr_prs_lr.append(tr_pr)
    va_prs_lr.append(va_pr)
    te_prs_lr.append(te_pr)


    # plot precision for tr, va, and te
    plt.plot(range(1,7), tr_prs_lr, label="Train")
    plt.plot(range(1,7), va_prs_lr, label="Validation")
    plt.plot(range(1,7), te_prs_lr, label="Test")
    plt.xticks(np.arange(1, 6.2, 1))
    plt.yticks(np.arange(0.74, 0.81, .01))
    plt.title("Precision for Logistic Regression")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Maximum degree")
    plt.savefig("LRprec.pdf")


## We try Logistic Regression on tx for polynomial basis of different degrees
analyze_logistic_regression(x_1, Y)
