# -*- coding: utf-8 -*-
# Maximilian Christ (max.christ@me.com)

"""
Finally I found a fast dtw implementation in C with correct python bindings and not a hack with the ucr time series subsequence search.

pip install git+https://github.com/lukauskas/mlpy-plus-dtw
    or
pit install git+https://github.com/MaxBenChrist/mlpy-plus-dtw

This is an improved version of the DTW metric implementend in the mlpy packge by User Saulius Lukauskas.
Unfortunately, it seems that the mlpy package is not actively developed anymore.
(the latest version 3.5.0 was released in 2012)

"""

import time
import pandas as pd
import numpy as np

from itertools import product

from mlpy.dtw import dtw_std
from scipy.stats import randint

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

EOTS = -9999
# EOTS stands for "end of time series", which is our faked np.NaN. sklearn will not pass arrays with np.NaN because of
# its check_array method, but those those np.NaNs are needed for variable sized time series.
# So, for the usage of the sklearn nearest neighbour search we need a faked NaN


def _finite_of(x):
    """
    Removes all values from x that are not equal to EOTS

    :param x: the input
    :type x: iterable
    :return: the cleaned version of x
    :rtype: numpy.array

    """
    x = np.asarray(x)
    return x[x != EOTS]

def construct_kNN_Regressor(k, warping_penalty, constraint="slanted_band"):
    """
    Constructs the kNN Regressor under a DTW metric
    """
    dtw_metric = lambda x, y: dtw_std(x, y,
                                      dist_only=True,
                                      constraint=constraint,
                                      k=k,
                                      warping_penalty=warping_penalty)

    reg = KNeighborsRegressor(n_neighbours=5,
                              metric=dtw_metric,
                              n_jobs=1)

    return reg

def construct_X_from_tsfresh_container(df, column_id="id", column_sort="sort", column_value="value",
                                       all_possible_timestamps=None):
    """
    Constructs the feature matrix for the kNN Regressor under a DTW metric. The time series container should be in flat
    format

    You want to call this method differently for train and test set. However, it could be that for some time stamps,
    only readings are available in one of the sets. For this, we have the all_possible_timestamps iterable.
    Just collect all possible time stamps for sensor recordings from both train and test set and pass it as this
    parameter.

    """

    X = df.pivot(index=column_id, columns=column_sort, values=column_value)

    if all_possible_timestamps is not None:
        new_cols = list(set(all_possible_timestamps) - set(X.columns))
        X = pd.concat([df, pd.DataFrame(columns=new_cols, index=X.index)], axis=1)

    X = X.fillna(EOTS)

    return X



# todo: clean and refactor the following code

#
# def predict_kNNdtwReg(est, df, index, timestamps=None):
#     df = df.pivot(index="id", columns="sort", values="value")
#
#     if timestamps is not None:
#         new_cols = list(set(timestamps) - set(df.columns))
#         df = pd.concat([df, pd.DataFrame(columns=new_cols, index=df.index)], axis=1)
#
#     df = df.fillna(EOTS)
#
#     # make sure predictions are in right order
#     df = df.loc[index, :]
#     return est.predict(df.values)
#
#
# def random_gridsearch_kNNdtwReg(df, y, n_iter=5, timestamps=None):
#     """
#     df should be time series in tsfresh format
#     y the target vector
#     """
#
#     df = df.pivot(index="id", columns="sort", values="value")
#
#     if timestamps is not None:
#         new_cols = list(set(timestamps) - set(df.columns))
#         df = pd.concat([df, pd.DataFrame(columns=new_cols, index=df.index)], axis=1)
#
#     df = df.fillna(EOTS)
#
#     # specify parameters and distributions to sample from
#     param_dist = {"n_neighbors": randint(1, 10),
#                   "weights": ["uniform", "distance"],
#                   "metric_params": [{"k": k, "warping_penalty": wp} for k, wp in product([1, 3, 5, 10, 15, 20],
#                                                                                          [0, .1, .25, .5, .75])]
#                   }
#
#     reg = KNeighborsRegressor(metric=fdtw)
#     random_search = RandomizedSearchCV(reg,
#                                        param_distributions=param_dist,
#                                        n_iter=n_iter,
#                                        verbose=2,
#                                        n_jobs=30,
#                                        error_score=9999)
#
#     start = time.time()
#     random_search.fit(df.values, y.loc[df.index].values)
#     end = time.time()
#
#     random_search.fitting_time = end-start
#
#     return random_search
#
#
# def fit_dtw_pipe(df, y, timestamps=None):
#
#     df = df.pivot(index="id", columns="sort", values="value")
#     if timestamps is not None:
#         new_cols = list(set(timestamps) - set(df.columns))
#         df = pd.concat([df, pd.DataFrame(columns=new_cols, index=df.index)], axis=1)
#     df = df.fillna(EOTS)
#
#     pipe = Pipeline([("kNN_dtw",  KNeighborsRegressor(n_neighbors=3,
#                                                       weights="distance",
#                                                       metric=fdtw,
#                                                       n_jobs=8,
#                                                       metric_params={"k": 10,
#                                                                      "warping_penalty": 0.1}
#                                                       ))])
#
#     start = time.time()
#     pipe.fit(df.values, y.loc[df.index].values)
#     end = time.time()
#
#     pipe.fitting_time = end - start
#
#     return pipe
