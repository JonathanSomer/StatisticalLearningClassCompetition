import numpy as np
from sklearn.linear_model import Ridge
from regressors.base_regressor import BaseRegressor
from data_pre_processing.clean_data import remove_bad_movies
from data_pre_processing.fill_missing_values import fill_ratings_with_mean_per_user

class FillWithMeanPerUserCleanMoviesRidgeRegressor(BaseRegressor):

    def __init__(self, alpha):
        self.alpha = alpha
        self._reg = Ridge(alpha = alpha)
        self.bad_movie_indexes = None

    def fit(self, X, Y):
        assert len(X.shape) == 3
        X = self._prepare_X(X)
        self._reg = self._reg.fit(X, Y)

    def predict(self, X):
        assert len(X.shape) == 3
        X = self._prepare_X(X)
        return self._reg.predict(X)

    def __str__(self):
        return "Fill With Mean Per User + Clean Movies + Ridge {}".format(self.alpha)


    def _prepare_X(self, X_raw):
        X_raw, self.bad_movie_indexes = remove_bad_movies(X_raw, self.bad_movie_indexes)
        X, _ = fill_ratings_with_mean_per_user(X_raw)
        X = X[:,1,:] # only ratings

        return X.reshape(X.shape[0], -1)
