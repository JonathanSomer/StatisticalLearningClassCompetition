import numpy as np
from sklearn.linear_model import LinearRegression
from regressors.base_regressor import BaseRegressor
from data_pre_processing.fill_missing_values import fill_ratings_with_mean_per_user


class NCARegressor(BaseRegressor):

    def __init__(self):
        self._reg = LinearRegression()
        self.bad_movie_indexes = None


    def fit(self, X, Y):
        assert len(X.shape) == 3
        X = self._prepare_X(X)
        
        self._reg = self._reg.fit(X, Y[:, 1])

    def predict(self, X):
        assert len(X.shape) == 3
        X = self._prepare_X(X)
        
        return self._reg.predict(X)

    def __str__(self):
        return "Neighbor Component Analysis + KNN"


    def _prepare_X(self, X_from_input):
        X, _ = fill_ratings_with_mean_per_user(X_from_input)
        X = X[:,1,:] # only ratings

        if self.bad_movie_indexes is None:
            n_users_didnt_see_movie = np.sum(np.isnan(X_from_input[:,1,:]), axis=0)
            self.bad_movie_indexes = np.argwhere(n_users_didnt_see_movie > 2500).squeeze()

        X = np.delete(X, self.bad_movie_indexes, axis = 1)
        return X.reshape(X.shape[0], -1)
