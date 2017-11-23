General ideas:
-----------------------------

Basically, use http://surpriselib.com/

* Baselines
    * Simpler ones -> Need to try with `surprise.prediction_algorithms.baseline_only.BaselineOnly(bsl_options={})`
        * Global mean :)
        * User mean   :)
        * Item mean   :)
    * Aditional
        * User rating depends on number of ratings
        * User rating depends on overall rating for the item

-----------------------------

From here on, use regularization! Better Ridge, maybe Lasso.

* Neighborhood models (http://surprise.readthedocs.io/en/stable/knn_inspired.html). In ADA we mentioned K-means++ to optimize the first cluster center at random from the data points.
    * Find sets of similar users
    * Find sets of similar items
    * Correlation/Cosine similarity suggested in post (one for users, one for items)
* Matrix factorization. For sparse matrices. Non-negative elements. Missing elements are not the same as elements equal to 0. (!!!)
    * Standard SVD `surprise.prediction_algorithms.matrix_factorization.SVD`
    * Asymmetric SVD
    * SVD++ `surprise.prediction_algorithms.matrix_factorization.SVDpp`
    * NMF (already implemented, but not properly?) `surprise.prediction_algorithms.matrix_factorization.NMF`

-----------------------------

Put all together...

Ensemble methods (http://scikit-learn.org/stable/modules/ensemble.html)
* Linear regression
* Gradient boosted decision trees - can apply different methods to different slices of data! We can cluster by: (!!!)
    * Number of items rated
    * Number of users that rated the item
    * Factor vectors of users and items (?)
* Maybe just manually cluster users and items depending on the number of ratings, and check the method that works best for each? Note: We'd train on the entire dataset for both cases I think