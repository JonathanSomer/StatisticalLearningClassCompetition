import numpy as np
from sklearn.linear_model import Ridge
from regressors.base_regressor import BaseRegressor
from data_pre_processing.clean_data import remove_bad_movies
from data_pre_processing.fill_missing_values import fill_ratings_with_mean_per_user
from regressors.fill_with_mean_per_user_clean_movies_ridge_flat_regressor import FillWithMeanPerUserCleanMoviesRidgeFlatRegressor


class UserMovieBiasRegressor(BaseRegressor):

    def __init__(self, alpha=10, minim=1.1, maxim=4.9):
        self.user_movie_interaction_regressor = FillWithMeanPerUserCleanMoviesRidgeFlatRegressor(maxim = 4.7, minim = 1.3)


    def fit(self, X, Y):
        assert len(X.shape) == 3
        self.user_movie_interaction_regressor.fit(X,Y)
        
        X = self._prepare_X(X)
        self._reg = self._reg.fit(X, Y)

    def predict(self, X):
        assert len(X.shape) == 3
        X = self._prepare_X(X)
        predictions = self._reg.predict(X)
        predictions[predictions > self.maxim] = self.maxim
        predictions[predictions < self.minim] = self.minim
        return predictions

    def __str__(self):
        return "FLAT + Ridge, alpha {}, min {}, max {}".format(self.alpha, self.minim, self.maxim)


    def _prepare_X(self, X_raw):
        X_raw, self.bad_movie_indexes = remove_bad_movies(X_raw, self.bad_movie_indexes)
        X, _ = fill_ratings_with_mean_per_user(X_raw)
        X = X[:,1,:] # only ratings

        return X.reshape(X.shape[0], -1)
