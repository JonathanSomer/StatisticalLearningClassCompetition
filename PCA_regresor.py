import numpy as np
from sklearn.decomposition import PCA


class PCAOnlyRatingsRegresor(object):
    def __init__(self, number_of_componnent, regresor):
        self._number_of_componnent = number_of_componnent
        self._pca = PCA(n_components=number_of_componnent)
        self._reg = regresor

    def fit(self, X, Y):
        assert len(X.shape) == 3
        assert X.shape[2] > self._number_of_componnent
        X = X[:, 1, :]
        Y = Y[:, 1]
        X = self._pca.fit_transform(X, Y)
        self._reg.fit(X, Y)

    def predict(self, X, Y):
        assert len(X.shape) == 3
        assert X.shape[2] > self._number_of_componnent
        X = X[:, 1, :]
        X = self._pca.transform(X)
        return self._reg.predict(X)
