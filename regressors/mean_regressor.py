import numpy as np
from regressors.base_regressor import BaseRegressor


class MeanRegressor(BaseRegressor):

    def fit(self, X, Y):
        pass

    def predict(self, X):
        X = np.nan_to_num(X)
		assert(len(X.shape) == 3)
		res = np.zeros(len(X))
		for i in range(len(X)):
			cur = X[i,1,:]
			cur = cur[cur!=0]
			res[i] = np.mean(cur)
		return res
	
	def __str__(self):
		return 'just_guess_mean'