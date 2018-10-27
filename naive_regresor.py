import numpy as np
from sklearn.linear_model import LinearRegression


class NaiveRegresor(object):

    def __init__(self):
        self._reg = LinearRegression()

    def fit(self, X, Y):
        assert len(X.shape) == 3
        X = X.reshape(X.shape[0], -1)
        self._reg = self._reg.fit(X, Y)

    def predict(self, X):
        return self._reg.predict(X.reshape(X.shape[0], -1))
