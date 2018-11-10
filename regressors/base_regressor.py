from abc import abstractmethod

# This is the base class all regressors should inherit from
class BaseRegressor(object):

    @abstractmethod
    def fit(self, X_train, Y_train):
        raise NotImplementedError("Must implement a fit method")

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError("Must implement a predict method")

    @abstractmethod # This should be an unique name, later used in the benchmark table
    def __str__(self):
    	return "Base Regressor"
