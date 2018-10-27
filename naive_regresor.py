import numpy as np
from sklearn.linear_model import LinearRegression

class NaiveRegresor(object):

    def __init__(self):
        self._reg = LinearRegression()

    def fit(self, X, Y):
        self._reg = self._reg.fit(X, y)
    
    def predict(self, X):
        return self._reg.predict(X)