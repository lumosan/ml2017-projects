import numpy as np
import scipy.sparse as sp
import csv


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()

def deal_line(line):
    pos, rating = line.split(',')
    row, col = pos.split("_")
    row = row.replace("r", "")
    col = col.replace("c", "")
    return int(row), int(col), float(rating)

def load_data(data_path):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(data_path)[1:]
    return preprocess_data(data)

def preprocess_data(data):
    """Preprocessing the text data, conversion to numerical array format."""
    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings.tocsr()

def generate_surprise_input_csv(filename='new_file', data_path='', output_path='output_'):
    """Reads the text data and outputs it in a file with `surprise` format"""
    def write(parsed_list):
        with open('{dp}.csv'.format(dp=write_path), 'w') as csvfile:
            fieldnames = ['item', 'user', 'rating']
            writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
            for (i, u, r) in parsed_list:
                writer.writerow({'item':i,'user':u,'rating':r})
    read_path = '{dp}{f}.csv'.format(dp=data_path, f=filename)
    write_path = '{odp}{f}'.format(odp=output_path, f=filename)
    data = read_txt(read_path)[1:]
    parsed_data = [deal_line(line) for line in data]
    write(parsed_data)


def save_csv(data_sp, prediction_path='', filename='new_file'):
    """Given a csr sparse matrix `data_sp` writes a Kaggle-style csv file"""
    with open('{dp}{fn}.csv'.format(dp=prediction_path, fn=filename), 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        # Get non-zero elements
        (rows, cols, vals) = sp.find(data_sp)
        for (i, u, v) in zip(rows, cols, vals):
            writer.writerow({'Id':'r{r}_c{c}'.format(r=i+1,c=u+1),'Prediction':v})


#TODO: See if I can simplify these two functions...
def save_csv_rec(data_rec, pred_rec, prediction_path='',
    filename='new_file'):
    """Given an array `data_rec` and a vector of predictions
    `pred_rec` in the format required by library 'recommend',
    writes a Kaggle-style csv file
    """
    with open('{dp}{fn}.csv'.format(dp=prediction_path, fn=filename), 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for e in range(data_rec.shape[0]):
            writer.writerow({'Id':'r{r}_c{c}'.format(r=data_rec[e,1]+1,
                c=data_rec[e,0]+1),'Prediction':pred_rec[e]})


def split_data(ratings, min_num_ratings, p_test=0.1, verbose=False, rnd_seed=998):
    """Split the ratings into training data and test data.
    Args:
        min_num_ratings:
            all users and items we keep must have at least min_num_ratings per user and per item.
    """
    # set seed
    np.random.seed(rnd_seed)

    # change sp matrix to lil
    ratings = ratings.tolil()

    # compute number of ratings per user and item
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()

    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][:, valid_users]

    # LIL is a convenient format for constructing sparse matrices
    train = sp.lil_matrix(valid_ratings.shape)
    test = sp.lil_matrix(valid_ratings.shape)

    valid_ratings_i, valid_ratings_u, valid_ratings_v = sp.find(valid_ratings)
    valid_ratings_p_idx = np.random.permutation(range(len(valid_ratings_i)))

    n_test = int(p_test*len(valid_ratings_i))

    for idx in valid_ratings_p_idx[:n_test]:
        test[valid_ratings_i[idx], valid_ratings_u[idx]] = valid_ratings_v[idx]

    for idx in valid_ratings_p_idx[n_test:]:
        train[valid_ratings_i[idx], valid_ratings_u[idx]] = valid_ratings_v[idx]

    if verbose:
        print("Total number of nonzero elements in original data: {v}".format(v=ratings.nnz))
        print("Total number of nonzero elements in train data:    {v}".format(v=train.nnz))
        print("Total number of nonzero elements in test data:     {v}".format(v=test.nnz))

    # convert to CSR for faster operations
    return valid_ratings.tocsr(), train.tocsr(), test.tocsr()
