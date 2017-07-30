# Fast-Parallel-DTW-kNN-Python

A k-Nearest-Neighbour Search under the Dynamic Time Warping Metric is often in
the literature reported to achieve the highest accuracies.

However, the runtime costs are quite high, so an efficient implementation is key.

I compared different setups and implementations that can be used from Python.
This repo contains the best combinations that I came up with.

It used an enhanced DTW C implementation and the kNN algorithm from sklearn
which is running parallel.
