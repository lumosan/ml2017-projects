import os
import csv
import scipy.sparse as sp
from datafile_methods.data_processing import load_data, k_fold_split, parse_data_sur
from surprise.dataset import Reader
from surprise import Dataset


## Methods for loading and preparing data
def prepare_data(k=5, data_path=''):
    """Splits data for training and test and for Surprise library"""
    # Load training dataset and example submission
    ratings = load_data('{dp}data_train.csv'.format(dp=data_path))
    sample_submission = load_data('{dp}sample_submission.csv'.format(dp=data_path))
    # Split training data for cross-validation and save files
    folds = k_fold_split(ratings, k=k)
    for i, f in enumerate(folds):
        save_csv(f, prediction_path=data_path, filename='fold{}'.format(i))

    # Parse input data files into surprise format
    # initialize lists with names of files
    train_sub_filenames = ['data_train', 'sample_submission']
    fold_filenames = ['fold{}'.format(i) for i in range(k)]
    # save files
    for fn in train_sub_filenames:
        parse_data_sur([fn], data_path=data_path, output_fn=fn,
            output_path=data_path+'surprise/')

    # prepare and save fold files
    for i in range(k):
        train_fn = fold_filenames.copy()
        test_fn = train_fn.pop(i)
        parse_data_sur(train_fn, data_path=data_path,
            output_fn='fold_tr_{}'.format(i),
            output_path=data_path+'surprise/')

        parse_data_sur([test_fn], data_path=data_path,
            output_fn='fold_te_{}'.format(i),
            output_path=data_path+'surprise/')
    return folds, ratings, sample_submission

def load_datasets_sur(data_path='../data/surprise/'):
    """Load all datasets for 'surprise' library"""
    # Define paths to dataset files
    folds_tr_dp = [os.path.expanduser('{}fold_tr_{}.csv'.format(data_path, i)) for i in range(5)]
    folds_te_dp = [os.path.expanduser('{}fold_te_{}.csv'.format(data_path, i)) for i in range(5)]
    rat_dp = os.path.expanduser('{}data_train.csv'.format(data_path))
    sub_dp = os.path.expanduser('{}sample_submission.csv'.format(data_path))

    # Define a Reader
    reader = Reader(line_format='item user rating', sep=',')

    # Load datasets
    folds_tr_ds = [Dataset.load_from_file(f_dp, reader=reader) for f_dp in folds_tr_dp]
    folds_te_ds = [Dataset.load_from_file(f_dp, reader=reader) for f_dp in folds_te_dp]
    ratings_ds = Dataset.load_from_file(rat_dp, reader=reader)
    sample_submission_ds = Dataset.load_from_file(sub_dp, reader=reader)

    # Retrieve trainsets
    folds_tr = [f_ds.build_full_trainset() for f_ds in folds_tr_ds]
    folds_te = [f_ds.build_full_trainset() for f_ds in folds_te_ds]
    ratings = ratings_ds.build_full_trainset()
    sample_submission = sample_submission_ds.build_full_trainset()
    return folds_tr, folds_te, ratings, sample_submission


## Methods for creating prediction Kaggle-style csv files
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

def save_csv_sur(data_pred, prediction_path='', filename='new_file'):
    """Given a list `data_pred` in the format generated by library
    'surprise', writes a Kaggle-style csv file
    """
    with open('{dp}{fn}.csv'.format(dp=prediction_path, fn=filename), 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for p in data_pred:
            writer.writerow({'Id':'r{r}_c{c}'.format(r=p.iid, c=p.uid),
                             'Prediction':p.est})
