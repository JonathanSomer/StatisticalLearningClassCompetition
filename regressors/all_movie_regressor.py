import numpy as np
import pandas as pd
from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
from auto_ml.utils_models import load_ml_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from regressors.base_regressor import BaseRegressor
from data_pre_processing.clean_data import remove_bad_movies
from data_pre_processing.fill_missing_values import fill_ratings_with_mean_per_user
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras import optimizers
from keras import callbacks
from keras import regularizers
from keras import backend
from autokeras import DeepSupervised
from autokeras.nn.loss_function import regression_loss

class AllMovieRegressor(BaseRegressor):
	def __init__(self, model = "auto_ml", dl_n_epochs = 20, pca_n_components = 20, verbose = False, rf_n_jobs = 16, rf_n_estimetors = 100):
		self._reg = None
		self.bad_movie_indexes = None
		self.representation_creator = PerUserMovie_RepresentationCreator(n_components = pca_n_components)
		self.scaler = MinMaxScaler()
		self.n_epochs = dl_n_epochs
		self.rf_n_jobs = rf_n_jobs
		self.rf_n_estimetors = rf_n_estimetors
		self.verbose = verbose
		self.model = model
		
		def preprocess_none(X_predict):
			return X_predict
		
		self.preprocessing_of_model = preprocess_none
		
	
	def fit(self, X, Y):
		assert len(X.shape) == 3
		X = self._prepare_X(X, Y, train = True)
		
		X, Y = self.representation_creator.fit_transform_for_each_movie(X, Y)
		
		rand_perm = np.random.permutation(len(X))
		
		X = X[rand_perm]
		Y = Y[rand_perm]
		
		print()
		print(X.shape)
		
		self._reg = self._model(X, Y)

	def predict(self, X):
		assert len(X.shape) == 3
		X = self._prepare_X(X)
		X = self.representation_creator.transform_for_target_movie_only(X)
		X = self.preprocessing_of_model(X)
		return self._reg.predict(X)

	def __str__(self):
		return "PcaAllMoviesRegressor\nido"
	
	def _prepare_X(self, X_raw, Y=None, train = False):
		X_raw, self.bad_movie_indexes = remove_bad_movies(X_raw, self.bad_movie_indexes)
		X, _ = fill_ratings_with_mean_per_user(X_raw)
		X = X[:,1,:]
		
		return X
	
	def rmse(self, y_true, y_pred):
		return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

	def _model(self, X_train, Y_train):
		if self.model == "random forest" or self.model == "rf":
			return self._model_rf(X_train, Y_train)
		elif self.model == "deep learning" or self.model == "dl":
			return self._model_dl(X_train, Y_train)
		elif self.model == "auto_ml" or self.model == "automl" or self.model == "automl":
			return self._model_auto_ml(X_train, Y_train)
		elif self.model == "auto_keras" or self.model == "autokeras" or self.model == "autokr":
			return self._model_auto_keras(X_train, Y_train)
		else:
			raise "Did not find {} model".format(self.model)

	def _model_dl(self, X_train, Y_train):
		model = Sequential()
		model.add(Dense(100, kernel_initializer='glorot_uniform', activation='relu'))
		model.add(Dense(10, kernel_initializer='glorot_uniform', activation='sigmoid'))
		model.add(Dense(1, kernel_initializer='glorot_uniform'))
		model.compile(loss='mse', optimizer='adam', metrics=[self.rmse])
		model.fit(X_train, Y_train, epochs = self.n_epochs, verbose=self.verbose)
		return model
		
	def _model_rf(self, X_train, Y_train):
		model = RandomForestRegressor(n_estimators = self.rf_n_estimetors, verbose=self.verbose, n_jobs=self.rf_n_jobs)
		model.fit(X_train, Y_train)
		return model

	def _model_auto_ml(self, X_train, Y_train):
		column_descriptions = {'OUT': 'output'}
		model = Predictor(type_of_estimator='regressor',
									  column_descriptions=column_descriptions, verbose=self.verbose)
		X = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))
		X['OUT'] = Y_train

		model.train(X)
		def preprocess_to_pandas(X_predict):
			return pd.DataFrame(X_predict.reshape(X_predict.shape[0], -1))
		
		self.preprocessing_of_model = preprocess_to_pandas
		return model
	
	def _model_auto_keras(self, X_train, Y_train):
		model = AutoKerasRegressor(verbose = self.verbose)
		model.fit(X_train, Y_train)
		return model
		
def AutoKerasRegressor(DeepSupervised):
	
	def __init__(self, vebose = False):
		super().__init__(verbose = verbose)
	
	@property
	def loss(self):
		return regression_loss
		

class PerUserMovie_RepresentationCreator():
	def __init__(self, n_components):
		self.n_components = n_components
		self.pca = PCA(n_components = self.n_components)
		self.n_movie = None
		
	def fit_transform_for_each_movie(self, X, Y):
		self.n_movie = X.shape[1]
		
		assert(len(X.shape) == 2)
		assert(len(Y.shape) == 1)
		
		X_pca = self.pca.fit_transform(X)
		
		X_all = np.zeros((X.shape[0], X.shape[1] + 1))
		X_all[:,:X.shape[1]] = X
		X_all[:,X.shape[1]] = Y
		X = X_all
			
		self._calculate_means_per_movies(X, X_pca)
		
		resX = []
		resY = []
		for imovie in range(len(X[0])):
			for iuser in range(len(X)):
				resX += [self._get_user_representation_for_movie(iuser, imovie, X, X_pca)]
				resY += [X[iuser, imovie]]
				
		return np.vstack(resX), np.array(resY)
					
	def transform_for_target_movie_only(self, X):
		assert(len(X.shape) == 2)
		assert(X.shape[1] == self.n_movie)
		X_pca = self.pca.transform(X)
		return self._get_all_users_representation_for_single_movie(self.n_movie, X, X_pca)
		
	def _get_all_users_representation_for_single_movie(self, imovie, X, X_pca):
		res = []
		
		for iuser in range(len(X)):
			res += [self._get_user_representation_for_movie(iuser, imovie, X, X_pca)]
		
		return np.vstack(res)
	
	def _get_user_representation_for_movie(self, iuser, imovie, X, X_pca):
		positive = 0
		negative = 1
		mean_movies = self.mean_per_component_sign_movie[:,:,imovie]
		
		res = np.zeros(self.n_components * 2 + 1)
		res[:self.n_components] = X_pca[iuser]
		for icomponnent in range(self.n_components):
			current_mean = 0
			if X_pca[iuser, icomponnent] > 0:
				current_mean = mean_movies[icomponnent, positive] 
			elif X_pca[iuser, icomponnent] < 0:
				current_mean = mean_movies[icomponnent, negative]
			res[self.n_components + icomponnent] = current_mean
		res[-1] = np.mean(X[iuser])
		return res

	def _calculate_means_per_movies(self, X, X_pca):
		n_movies = X.shape[1]
		self.mean_per_component_sign_movie = np.zeros([self.n_components, 2, n_movies])
		
		positive = 0
		negative = 1
		
		for icomponnent in range(self.n_components):
			X_componnent = X_pca[:,icomponnent]
			mask_positive = X_componnent > 0
			mask_negative = X_componnent < 0
			
			for imovie in range(n_movies):
				scores_current_X = X[:, imovie]
				self.mean_per_component_sign_movie[icomponnent,positive,imovie] = np.average(scores_current_X[mask_positive], weights = X_componnent[mask_positive])
				self.mean_per_component_sign_movie[icomponnent,negative,imovie] = np.average(scores_current_X[mask_negative], weights = X_componnent[mask_negative])	