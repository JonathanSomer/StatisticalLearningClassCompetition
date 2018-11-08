import numpy as np
from sklearn.linear_model import LinearRegression

class NaiveRegressorDateSubstraction(object):

    def __init__(self, movie_dates, output_movie_date):
        self._movie_dates = movie_dates.reshape(1, -1)
        self._output_movie_date = output_movie_date
        self._reg = NaiveRegresor()

    def _clean_data(self, X, Y):
        assert self._movie_dates.shape[-1] == X.shape[-1]
        X[:, 0, :] -= self._movie_dates
        if len(Y.shape) == 1:
            Y -= self._output_movie_date
        else:
            Y[:, 0] -= self._output_movie_date
        return X, Y

    def fit(self, X, Y):
        assert len(X.shape) == 3
        X, Y = self._clean_data(X, Y)
        self._reg = self._reg.fit(X, Y)
        return self

    def predict(self, X, Y):
        assert len(X.shape) == 3
        X, Y = self._clean_data(X, Y)
        return self._reg.predict(X, Y)
