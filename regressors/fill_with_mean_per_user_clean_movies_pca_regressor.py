import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from regressors.base_regressor import BaseRegressor
from data_pre_processing.clean_data import remove_bad_movies
from data_pre_processing.fill_missing_values import fill_ratings_with_mean_per_user
from sklearn.ensemble import RandomForestRegressor

class FillWithMeanPerUserCleanMoviesPCARegressor(BaseRegressor):

    def __init__(self):
        self._reg = LinearRegression()
        self.bad_movie_indexes = None
        self.scaler = MinMaxScaler()
        self.pca = None

    def fit(self, X, Y):
        assert len(X.shape) == 3
        X = self._prepare_X(X, train = True)
        self._reg = self._reg.fit(X, Y[:, 1])

    def predict(self, X):
        assert len(X.shape) == 3
        X = self._prepare_X(X)
        return self._reg.predict(X)

    def __str__(self):
        return "Fill With Mean Per User + Clean Movies + PCA"


    def _prepare_X(self, X_raw, train = False):
        X_raw, self.bad_movie_indexes = remove_bad_movies(X_raw, self.bad_movie_indexes)
        X, _ = fill_ratings_with_mean_per_user(X_raw)
        X = X[:,1,:] # only ratings

        if train:
            self.scaler.fit(X)
        X_rescaled = self.scaler.transform(X)

        if train:
            self.pca = PCA(n_components = 75).fit(X_rescaled)
        X_components = self.pca.transform(X_rescaled)

        return X_components.reshape(X.shape[0], -1)
