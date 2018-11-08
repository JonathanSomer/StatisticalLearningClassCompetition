import numpy as np
import pandas as pd


def fill_with_mean(X):
    assert len(X.shape) == 3 and X.shape[1] == 2
    X = X.reshape(X.shape[0], -1)
    X = pd.DataFrame(X)
    X = X.fillna(X.mean())
    X = np.array(X)
    return X.reshape(X.shape[0], 2, -1)


def fill_with_zeros(X):
    return np.nan_to_num(X)


def fill_with_num(X, num):
    filled_matrix = fill_with_zeros(X)
    filled_matrix[filled_matrix == 0] = num
    return filled_matrix
