import numpy as np
import pandas as pd


def fill_with_mean(X):
    assert len(X.shape) == 3, 'expect X to be 3d array'
    s2 = X.shape[1]
    X = X.reshape(X.shape[0], -1)
    X = pd.DataFrame(X)
    X = X.fillna(X.mean())
    X = np.array(X)
    return X.reshape(X.shape[0], s2, -1)
