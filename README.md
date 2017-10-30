# Project 1

In this project we developed the first ML concepts seen in class. We have done exploratory data analysis to understand the dataset and the
features, done feature processing and engineering to clean the dataset and extract more meaningful information,
implemented and use machine learning methods on real data, analyze the model and generate predictions using
those methods.

## Getting Started

These instructions will make sure the the files provided in a certain structure are understandable.

### File tree

The different files are organized in the tree structure shown beow. The two main folders separate the data from the code. It is necessary to have the CSV data files in the correct location for the script to compute them. The resulting submission file will also be released in this location.
The `scripts` folder contains the code. The `custum_methods` folder contains all the auxiliary functions the computation need.
At the end, the script is run thanks to the `run.py` file

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

## Running the script

Given the folder structure explained before, running the python script is straightforward. Using the ... just run the script as follow

```
> python run.py
```

## Deployment

Add additional notes about how to deploy this on a live system

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Lucía Montero Sanchis**
* **Nuno Mota Gonçalves**
* **Matteo Yann Feo**
