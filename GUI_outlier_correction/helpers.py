"""A few numpy helper functions.

Author: Romain Fayat, June 2021
"""
import numpy as np


def percentile(n):
    "Percentile function for agglomerating data."
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = f"percentile_{n:02d}"
    return percentile_


def get_intervals_idx(arr):
    "Determine the start and end idx of intervals of constant value in arr."
    changepoint = np.argwhere(arr[:-1] != arr[1:])
    intervals_start = np.append(0, changepoint.flatten())
    intervals_end = np.append(changepoint.flatten() + 1, len(arr))
    return intervals_start, intervals_end
