import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

class PcaRegressorNoDate(object):
    def __init__(self, number_of_componnent = 5, regresor = LinearRegression()):
        self._number_of_componnent = number_of_componnent
        self._pca = PCA(n_components=number_of_componnent)
        self._reg = regresor

    def fit(self, X, Y):
        assert len(X.shape) == 3
        assert X.shape[2] > self._number_of_componnent
        X = np.nan_to_num(X)
        X = X[:, 1, :]
        Y = Y[:, 1]
        X = self._pca.fit_transform(X, Y)
        self._reg.fit(X, Y)

    def predict(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] > self._number_of_componnent
        X = np.nan_to_num(X)
        X = X[:, 1, :]
        X = self._pca.transform(X)
        return self._reg.predict(X)

    def __str__(self):
        return "PCA Regressor No Date"
