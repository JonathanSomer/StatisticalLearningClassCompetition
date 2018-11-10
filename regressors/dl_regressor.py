import numpy as np
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

class DLRegressor(BaseRegressor):

    def __init__(self, n_epochs = 20):
        self._reg = None
        self.bad_movie_indexes = None
        self.scaler = MinMaxScaler()
        self.pca = None
        self.n_epochs = n_epochs

    def fit(self, X, Y):
        assert len(X.shape) == 3
        X = self._prepare_X(X, Y, train = True)
        self._reg = self._model(X)
        self._reg.fit(X, Y[:, 1], epochs = self.n_epochs)

    def predict(self, X):
        assert len(X.shape) == 3
        X = self._prepare_X(X)
        return self._reg.predict(X)

    def __str__(self):
        return "knn"


    def _prepare_X(self, X_raw, Y=None, train = False):
        X_raw, self.bad_movie_indexes = remove_bad_movies(X_raw, self.bad_movie_indexes)
        X, _ = fill_ratings_with_mean_per_user(X_raw)
        X = X[:,1,:] # only ratings

        # if train:
        #     self.scaler.fit(X)
        # X_rescaled = self.scaler.transform(X)

        # if train:
        #     self.pca = PCA(n_components = 75).fit(X_rescaled)
        # X_components = self.pca.transform(X_rescaled)
        
        # return X_components.reshape(X.shape[0], -1)
        return X
 
    def rmse(self, y_true, y_pred):
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    def _model(self, X_train):
        model = Sequential()
        model.add(Dense(100, kernel_regularizer=regularizers.l2(0.2), input_shape=X_train[0].shape))
        model.add(Activation('relu'))
        # model.add(Dropout(0.5, name='dropout_1'))
        model.add(Dense(100, kernel_regularizer=regularizers.l2(0.2)))
        model.add(Activation('relu'))
        model.add(Dense(10, kernel_regularizer=regularizers.l2(0.2)))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=[self.rmse])
        return model
