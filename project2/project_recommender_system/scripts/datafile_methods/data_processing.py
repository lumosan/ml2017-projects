import numpy as np
import scipy.sparse as sp
import csv


## Auxiliary methods for other methods in this script
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


## Data parsing and loading methods
def load_data(data_path):
    """Load data in text format (Kaggle-style csv file)
    Returns a sparse csr matrix with items as rows and users as cols
    """
    data = read_txt(data_path)[1:]
    return preprocess_data(data)

def parse_data_sur(filenames, data_path='', output_fn='new_file_sur',
    output_path='surprise_'):
    """Reads an input csv file and creates an equivalent one
    only with `surprise` format
    """
    def write(parsed_list):
        with open('{dp}.csv'.format(dp=write_path), 'w') as csvfile:
            fieldnames = ['item', 'user', 'rating']
            writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
            for (i, u, r) in parsed_list:
                writer.writerow({'item':i,'user':u,'rating':r})
    read_paths = ['{dp}{f}.csv'.format(dp=data_path, f=filename) for filename in filenames]
    write_path = '{odp}{f}'.format(odp=output_path, f=output_fn)
    data_list = [read_txt(read_path)[1:] for read_path in read_paths]
    parsed_data = [deal_line(line) for data in data_list for line in data]
    write(parsed_data)

## Method for splitting into training and test
def split_data(ratings, min_num_ratings=0, p_test=[.9, .1], rnd_seed=998):
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
    sets = [sp.lil_matrix(valid_ratings.shape) for p in p_test]

    # Get a random permutation of the valid ratings
    valid_ratings_i, valid_ratings_u, valid_ratings_v = sp.find(valid_ratings)
    valid_ratings_p_idx = np.random.permutation(range(len(valid_ratings_i)))

    # Get number of ratings in each set
    n_elem_sets = [int(p * len(valid_ratings_i)) for p in p_test]
    n_elem_sets[:0] = [0]
    n_sets = np.cumsum(n_elem_sets)
    # the last set contains all remaining data
    n_sets[-1] = len(valid_ratings_i)

    for s in range(len(sets)):
        for idx in valid_ratings_p_idx[n_sets[s]:n_sets[s+1]]:
            sets[s][valid_ratings_i[idx], valid_ratings_u[idx]] = valid_ratings_v[idx]

    # convert to CSR for faster operations
    return [s.tocsr() for s in sets]

def k_fold_split(ratings, k=5):
    """Uses function split_data to split training data to use for
    k-fold cross validation.
    """
    n_ratings = np.full((1,k),1/k).flatten()
    folds = split_data(ratings, p_test=n_ratings)
    return folds
