"""Strict local peaks and valleys of an input EWS series."""

import numpy as np
import pandas as pd


def detect(values, comparison_width):
    smooth = (
        pd.Series(np.asarray(values, float)).rolling(5, center=True, min_periods=5).mean().dropna()
    )
    peaks = []
    valleys = []
    k = comparison_width
    for i in range(k, len(smooth) - k):
        v = smooth.iloc[i]
        left = smooth.iloc[i - k : i]
        right = smooth.iloc[i + 1 : i + k + 1]
        if v > left.max() and v > right.max():
            peaks.append(int(smooth.index[i]))
        elif v < left.min() and v < right.min():
            valleys.append(int(smooth.index[i]))
    return {'peaks': peaks, 'valleys': valleys}
