import numpy as np
from sklearn.linear_model import LinearRegression
from regressors.base_regressor import BaseRegressor
from data_pre_processing.fill_missing_values import fill_ratings_with_mean_per_user

MAX_USERS_WHO_DIDNT_SEE_MOVIE = 2000 # discovered over cross validation
MAX_MOVIES_USER_DIDNT_SEE = 70

class FillWithMeanPerUserCleanMoviesCleanUsersRegressor(BaseRegressor):

    def __init__(self):
        self._reg = LinearRegression()
        self.bad_movie_indexes = None
        self.bad_user_indexes = None
        self.max_users_didnt_see_movie = MAX_USERS_WHO_DIDNT_SEE_MOVIE
        self.max_movies_user_didnt_see = MAX_MOVIES_USER_DIDNT_SEE


    def fit(self, X, Y):
        assert len(X.shape) == 3

        n_movies_user_didnt_see = np.sum(np.isnan(X[:,1,:]), axis=1)
        bad_user_indexes = np.argwhere(n_movies_user_didnt_see > self.max_movies_user_didnt_see).squeeze()

        X = np.delete(X, bad_user_indexes, axis = 0)
        Y = np.delete(Y, bad_user_indexes, axis = 0)
        
        X = self._prepare_X(X)
        self._reg = self._reg.fit(X, Y[:, 1])

    def predict(self, X):
        assert len(X.shape) == 3
        X = self._prepare_X(X, training = False)

        return self._reg.predict(X)

    def __str__(self):
        return "Fill With Mean Per User + Clean Movies + Clean Users {}".format(self.max_movies_user_didnt_see)


    def _prepare_X(self, X_from_input, training = True):
        X, _ = fill_ratings_with_mean_per_user(X_from_input)
        X = X[:,1,:] # only ratings

        if self.bad_movie_indexes is None:
            n_users_didnt_see_movie = np.sum(np.isnan(X_from_input[:,1,:]), axis=0)
            self.bad_movie_indexes = np.argwhere(n_users_didnt_see_movie > self.max_users_didnt_see_movie).squeeze()

        X = np.delete(X, self.bad_movie_indexes, axis = 1)

        return X.reshape(X.shape[0], -1)
