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
        * (What is contains)
    2. cost_gradient_methods.py
        * (What it contains)
    3. cross_validation_methods.py
        * (What is contains)
    4. data_processing_methods.py
        * (What it contains)
    5. proj1_helpers.py
        * (What is contains)
    6. standardizing_methods.py
        * (What it contains)

Along with `implementations.py`, that (What it contains), these scripts represent the dependencies of `run.py`.

## 3. Running the script

### 3.1. Pre-requisites

Before running the script make sure you do the following:

* Have `train.csv` and `test.csv` in the `data` directory
* Have Python installed and running on your machine

### 3.2. Running

    1. Open your terminal
    2. Navigate inside our `scripts` directory
    3. Run "python run.py"

Since our script doesn't receive any special arguments, the above should suffice to reproduce our results - and output them into a new file, `submission.csv`, automatically written to the `data` directory.

## 5. Authors

* **Lucía Montero Sanchis**
* **Nuno Mota Gonçalves**
* **Matteo Yann Feo**
