def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

def build_poly(x, degree, offset=True):
    """Polynomial basis functions for input data x,
    for up to a certain degree."""
    if offset:
        rows, cols = np.indices((x.shape[0], degree+1))
        tx = np.power(x[rows], cols)
    else:
        rows, cols = np.indices((x.shape[0], degree))
        tx = np.power(x[rows], cols+1)
    return tx

def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form. Uses build_poly."""
    return y, build_poly(x, 1, offset=True)

def split_data(x, y, ratio, seed=1):
    """Split the dataset based on the split ratio.

    E.g. if the ratio is 0.8 then 80% of the data set is dedicated
    to training (and the rest dedicated to testing)
    """
    np.random.seed(seed)

    size = int(ratio * x.shape[0])
    indices = np.random.permutation(x.shape[0])
    training_idx, test_idx = indices[:size], indices[size:]

    x_training, x_test = x[training_idx], x[test_idx]
    y_training, y_test = y[training_idx], y[test_idx]

    return x_training, x_test, y_training, y_test


def compute_mse(y, tx, w):
    """Calculates the loss using MSE."""
    if len(tx.shape) == 1:
        tx = tx.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    w = np.array(w).reshape(tx.shape[1], 1)
    z = y - tx.dot(w)
    z = z.T.dot(z)
    return z[0][0] / tx.shape[0]


# -------------------------------------------------------

#EX02
def compute_gradient(y, tx, w):
    """Computes the gradient."""
    return (tx.T.dot(tx.dot(w) - y)) / len(y)

#def gradient_descent(y, tx, initial_w, max_iters, gamma):




# Might have to move it to a different file?
def polynomial_regression():
    '''Constructing the polynomial basis function expansion of the data,
       and then running least squares regression.'''

    # define parameters
    degrees = [1, 3, 7, 12]

    # define structure of the figure
    num_row = 2
    num_col = 2
    f, axs = plt.subplots(num_row, num_col)

    for ind, degree in enumerate(degrees):
        x_standardized, mean_x, std_x = standardize(x)
        tx = build_poly(x_standardized, degree)
        w_opt, rmse = least_squares(y, tx)

        print("Processing {i}th experiment, degree={d}, rmse={loss}".format(
              i=ind + 1, d=degree, loss=rmse))

        # plot fit
        x_cont = np.linspace(np.min(x),np.max(x),100)
        x_cont_standardized = (x_cont-mean_x)/std_x
        X_cont = build_poly(x_cont_standardized, degree)
        y_cont = X_cont.dot(w_opt)
        plt.subplot(2,2,ind+1)
        plt.scatter(x, y, s=10),
        plt.plot(x_cont, y_cont, color='darkorange')

    plt.tight_layout()
    plt.savefig("visualize_polynomial_regression")
    plt.show()
