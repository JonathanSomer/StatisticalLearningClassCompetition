import numpy as np
from sklearn.linear_model import LinearRegression
from regressors.base_regressor import BaseRegressor
from data_pre_processing.fill_missing_values import fill_ratings_with_mean_per_user


class FillWithMeanPerUserRegressor(BaseRegressor):

    def __init__(self):
        self._reg = LinearRegression()


    def fit(self, X, Y):
        assert len(X.shape) == 3
        X = self._prepare_X(X)
        
        self._reg = self._reg.fit(X, Y)

    def predict(self, X):
        assert len(X.shape) == 3
        X = self._prepare_X(X)
        
        return self._reg.predict(X)

    def __str__(self):
        return "Fill With Mean Per User"


    def _prepare_X(self, X_from_input):
        X, _ = fill_ratings_with_mean_per_user(X_from_input)
        X = X[:,1,:] # only ratings
        return X.reshape(X.shape[0], -1)
