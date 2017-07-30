# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsRegressor

# Finally I found a fast dtw implementation in C with correct python bindings and not a hack with the
# ucr time series subsequence search
# pip install git+https://github.com/lukauskas/mlpy-plus-dtw

from mlpy.dtw import dtw_std
import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from itertools import product
import time
from sklearn.pipeline import Pipeline

""" from Documentation of ucrdtw method:
Arguments:
data - a list of floats or an ndarray for the time series to process
query - a list of floats or an ndarray for the time series to search for
warp_width - Allowed warp width as a fraction of query size
verbose - Optional boolean to print stats

Returns location, distance of the subquery

We use a hack to deploy the method. with two equal size time series it is actually calculating the dtw distance
"""

EOTS = -9999
# EOTS stands for "end of time series"
# EOTS is our faked np.NaN. sklearn will not pass arrays with np.NaN because of its check_array method
# but those those np.NaNs are needed for variable sized time series
# so, for the usage of the sklearn nearest neighbour search we need a faked NaN


def finite_of(x):
    x = np.asarray(x)
    return x[x != EOTS]


def fdtw(x, y, k, warping_penalty):
    return dtw_std(x, y, dist_only=True, constraint="slanted_band", k=k, warping_penalty=warping_penalty)


def predict_kNNdtwReg(est, df, index, timestamps=None):
    df = df.pivot(index="id", columns="sort", values="value")

    if timestamps is not None:
        new_cols = list(set(timestamps) - set(df.columns))
        df = pd.concat([df, pd.DataFrame(columns=new_cols, index=df.index)], axis=1)

    df = df.fillna(EOTS)

    # make sure predictions are in right order
    df = df.loc[index, :]
    return est.predict(df.values)


def random_gridsearch_kNNdtwReg(df, y, n_iter=5, timestamps=None):
    """
    df should be time series in tsfresh format
    y the target vector
    """

    df = df.pivot(index="id", columns="sort", values="value")

    if timestamps is not None:
        new_cols = list(set(timestamps) - set(df.columns))
        df = pd.concat([df, pd.DataFrame(columns=new_cols, index=df.index)], axis=1)

    df = df.fillna(EOTS)

    # specify parameters and distributions to sample from
    param_dist = {"n_neighbors": randint(1, 10),
                  "weights": ["uniform", "distance"],
                  "metric_params": [{"k": k, "warping_penalty": wp} for k, wp in product([1, 3, 5, 10, 15, 20],
                                                                                         [0, .1, .25, .5, .75])]
                  }

    reg = KNeighborsRegressor(metric=fdtw)
    random_search = RandomizedSearchCV(reg,
                                       param_distributions=param_dist,
                                       n_iter=n_iter,
                                       verbose=2,
                                       n_jobs=30,
                                       error_score=9999)

    start = time.time()
    random_search.fit(df.values, y.loc[df.index].values)
    end = time.time()

    random_search.fitting_time = end-start

    return random_search


def fit_dtw_pipe(df, y, timestamps=None):

    df = df.pivot(index="id", columns="sort", values="value")
    if timestamps is not None:
        new_cols = list(set(timestamps) - set(df.columns))
        df = pd.concat([df, pd.DataFrame(columns=new_cols, index=df.index)], axis=1)
    df = df.fillna(EOTS)

    pipe = Pipeline([("kNN_dtw",  KNeighborsRegressor(n_neighbors=3,
                                                      weights="distance",
                                                      metric=fdtw,
                                                      n_jobs=8,
                                                      metric_params={"k": 10,
                                                                     "warping_penalty": 0.1}
                                                      ))])

    start = time.time()
    pipe.fit(df.values, y.loc[df.index].values)
    end = time.time()

    pipe.fitting_time = end - start

    return pipe
