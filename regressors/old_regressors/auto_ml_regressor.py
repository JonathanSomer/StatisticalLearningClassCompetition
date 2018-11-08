# from auto_ml import Predictor
# from auto_ml.utils import get_boston_dataset
# from auto_ml.utils_models import load_ml_model
# import pandas as pd
# from regressors.base_regressor import BaseRegressor

# class AutoMlRegressor(BaseRegressor):

#     def __init__(self):
#         column_descriptions = {'OUT': 'output'}
#         self._my_auto_ml = Predictor(type_of_estimator='regressor',
#                                      column_descriptions=column_descriptions, verbose=False)

#     def fit(self, X, Y):
#         X = pd.DataFrame(X.reshape(X.shape[0], -1))
#         X['outdates'] = Y[:, 0]
#         X['OUT'] = Y[:, 1]
#         self._my_auto_ml.train(X)

#     def predict(self, X, Y):
#         X = pd.DataFrame(X.reshape(X.shape[0], -1))
#         X['outdates'] = Y
#         return self._my_auto_ml.predict(X)
