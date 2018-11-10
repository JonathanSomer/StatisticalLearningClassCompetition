import numpy as np


MAX_USERS_WHO_DIDNT_SEE_MOVIE = 0.333 # discovered over cross validation

def remove_bad_movies(X_raw, bad_movie_indexes = None):
	if bad_movie_indexes is None:
	    n_users_didnt_see_movie = np.sum(np.isnan(X_raw[:,1,:]), axis=0)
	    n = MAX_USERS_WHO_DIDNT_SEE_MOVIE * X_raw.shape[0]
	    bad_movie_indexes = np.argwhere(n_users_didnt_see_movie > n).squeeze()
	
	X = np.delete(X_raw, bad_movie_indexes, axis = 1)
	X = np.delete(X_raw, bad_movie_indexes, axis = 2)
	return X, bad_movie_indexes
