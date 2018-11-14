import numpy as np

def days_to_rate_target_movie(X_train, target_movie_release_date):
	return (X_train[:,2,0] - target_movie_release_date).reshape(X_train.shape[0], -1)

def diff_in_time_to_rate_target_movie(X_train, target_movie_release_date):
	all_users_dates = X_train[:,0,:]

	release_date_of_all = np.nanmin(all_users_dates, axis=0)

	time_after_release = all_users_dates - release_date_of_all
	no_nan_time_after_release = [a[~np.isnan(a)] for a in time_after_release]
	average_time_to_see_movie = [ np.mean(a) for a in no_nan_time_after_release]
	delt = target_movie_release_date - average_time_to_see_movie

	return delt.reshape(X_train.shape[0], -1)

# 
# def diff_in_time_to_rate_target_movie_only_5s(X_train, target_movie_release_date):
# 	return 

# def diff_in_time_to_rate_target_movie_only_4s(X_train, target_movie_release_date):
# 	return 
