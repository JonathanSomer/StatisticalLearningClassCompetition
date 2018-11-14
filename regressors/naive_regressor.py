import numpy as np
from sklearn.linear_model import LinearRegression
from regressors.base_regressor import BaseRegressor


class NaiveRegressor(BaseRegressor):

    def __init__(self):
        self._reg = LinearRegression()

    def fit(self, X, Y):
        assert len(X.shape) == 3
        X = np.nan_to_num(X)
        X = self._flatten_X(X)
        self._reg = self._reg.fit(X, Y)

    def predict(self, X):
        X = np.nan_to_num(X)
        X = self._flatten_X(X)
        return self._reg.predict(X)

    def __str__(self):
        return "NaiveRegressor"

    def _flatten_X(self, X):
        return X.reshape(X.shape[0], -1)[:,:199]
