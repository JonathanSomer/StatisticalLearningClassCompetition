import numpy as np
from sklearn.linear_model import LinearRegression
from regressors.base_regressor import BaseRegressor


class NaiveRegressor(BaseRegressor):

    def __init__(self):
        self._reg = LinearRegression()

    def fit(self, X, Y):
        assert len(X.shape) == 3
        X = np.nan_to_num(X)
        X = X.reshape(X.shape[0], -1)
        self._reg = self._reg.fit(X, Y[:, 1])

    def predict(self, X):
        X = np.nan_to_num(X)
        return self._reg.predict(X.reshape(X.shape[0], -1))

    def __str__(self):
        return "NaiveRegressor"
