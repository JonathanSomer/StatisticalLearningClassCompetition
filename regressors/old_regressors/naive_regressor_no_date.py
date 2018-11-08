# import numpy as np
# from sklearn.linear_model import LinearRegression
# from regressors.base_regressor import BaseRegressor

# class NaiveRegressorNoDate(BaseRegressor):
#     def __init__(self):
#         self._reg = LinearRegression()

#     def fit(self, X, Y):
#         assert len(X.shape) == 3
#         X = X[:, 1, :]
#         self._reg = self._reg.fit(X, Y[:, 1])
#         return self

#     def predict(self, X, Y):
#         X = X[:, 1, :]
#         return self._reg.predict(X)
