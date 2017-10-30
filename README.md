# Project #1

This README file explains the provided files' structure and hierarchy, along with an understandable description of their functionalities and use.

## 1. File hierarchy

The different files are organized like in the tree structure below. The two main folders, `scripts` and `data`, separate our code from our resources.

```
.
├── data
|   ├── test.csv
|   ├── train.csv
|   └── submission.csv
|
└── scripts
    ├── custom_methods
    |    ├── auxiliary_methods.py
    |    ├── cost_gradient_methods.py
    |    ├── cross_validation_methods.py
    |    ├── data_processing_methods.py
    |    ├── proj1_helpers.py
    |    └── standardizing_methods.py
    |
    ├── implementations.py
    └── run.py
```

## 2. Files' content

Each and every script contained in `custom_methods` is presented below:

    1. auxiliary_methods.py
        * sigmoid - Applies the sigmoid function.
        * get_batch - Generates a batch of a certain size, useful for stochastic gradient descent methods.
        * adaptive_lambda - Generates an adaptive learning rate that can be used for stochastic gradient descent methods instead of a constant one.

    2. cost_gradient_methods.py
        * compute_mse - Computes the mean squared error.
        * logistic_by_gd - Computes the loss and the model weights after one step of gradient descent for logistic regression.
        * reg_logistic_by_gd - The analogue of the previous function, in this case with a regularization term. It is possible to decide whether to penalize the offset term w0.

    3. cross_validation_methods.py
        * calculate_precision - Calculates precision for classification using labels {0, 1} or {-1, 1}.
        * split_data - Splits the data based on a split ratio.

    4. data_processing_methods.py
        * build_poly - Builds the polynomial basis functions for either a vector or the columns of a matrix.
        * build_model_data - Adds a first column of ones to include the offset in the data matrix.
        * PCA_analysis - Obtains a reduced set of features by applying principal component analysis.
        * build_poly_PCA - Computes a new set of features by combining polynomial basis and PCA.

    5. proj1_helpers.py
        * load_csv_data - Loads data.
        * predict_labels - Generates class prediction using labels {-1, 1}
        * predict_labels_bis - Generates class prediction using labels {0, 1}
        * create_csv_submission - Creates output file.

    6. standardizing_methods.py
        * standardize
        * standardize_test
        * standardize_ignoring_values
        * standardize_test_ignoring_values

Along with `implementations.py`, that contains the implementation of the learning methods seen in class, these scripts represent the dependencies of `run.py`.

## 3. Running the script

### 3.1. Pre-requisites

Before running the script make sure you do the following:

* Have `train.csv` and `test.csv` in the `data` directory
* Have Python 3 installed and running on your machine

### 3.2. Running

    1. Open your terminal
    2. Navigate inside our `scripts` directory
    3. Run "python run.py"

Since our script doesn't receive any special arguments, the above should suffice to reproduce our results - and output them into a new file, `submission.csv`, automatically written to the `data` directory.

## 4. Authors

* **Lucía Montero Sanchis**
* **Nuno Mota Gonçalves**
* **Matteo Yann Feo**
