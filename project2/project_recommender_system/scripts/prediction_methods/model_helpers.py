import numpy as np
from itertools import groupby
from sklearn.metrics import mean_squared_error

def compute_division(a, b):
    """Computes element by element division.
    If x/0 returns 0.
    """
    # Raises error if vectors have different lengths
    assert(len(a) == len(b))

    # Computes division
    res = a.copy()
    for i in range(len(a)):
        if b[i] == 0:
            res[i] = 0
        else:
            res[i] = a[i] / b[i]

    return res



def compute_error(data, u_features, i_features, nz):
    """Compute RMSE for prediction of nonzero elements."""
    preds = np.array([(u_features[:,u].dot(i_features[:,i]))
        for (i, u) in nz])
    vals = np.array([data[i,u] for (i,u) in nz])
    rmse = calculate_rmse(vals, preds)
    return rmse


def calculate_rmse(real_label, prediction):
    """calculate RMSE"""
    mse = mean_squared_error(real_label, prediction)
    rmse = np.sqrt(mse)
    return rmse


def build_index_groups(train):
    """Build groups for nnz rows and cols."""
    def group_by(data, index):
        """Group list of list by a specific index."""
        sorted_data = sorted(data, key=lambda x: x[index])
        groupby_data = groupby(sorted_data, lambda x: x[index])
        return groupby_data
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices