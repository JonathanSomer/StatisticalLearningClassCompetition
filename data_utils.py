import numpy as np
import os.path

DATA_BASE_DIR = './data'

TEST_X_DATES = 'test_dates_all.txt'
TEST_X_RATINGS = 'test_ratings_all.txt'
TEST_Y_DATES = 'test_y_date.txt'

TRAIN_X_DATES = 'train_dates_all.txt'
TRAIN_X_RATINGS = 'train_ratings_all.txt'
TRAIN_Y_DATES = 'train_y_date.txt'
TRAIN_Y_RATINGS = 'train_y_rating.txt'

NUMBER_OF_MOVIES = 99
# X: for 1000 users, 2 rows - first is the rating of 99 movies, second is the date for each rating
# Y: for 1000 users, rating and date for the "miss congeniality" film
def get_X_Y_train():
	X = []
	Y = []

	with open(_get_data_file_full_path_by_name(TRAIN_X_RATINGS)) as f:
		for line in f:
			movie_ratings = [int(rating) for rating in line.split()]
			assert len(movie_ratings) == NUMBER_OF_MOVIES
			X.append([movie_ratings])

	with open(_get_data_file_full_path_by_name(TRAIN_X_DATES)) as f:
		for line_number, line in enumerate(f):
			rating_dates = [int(rating) for rating in line.split()]
			assert len(rating_dates) == NUMBER_OF_MOVIES
			X[line_number].append(rating_dates)

	with open(_get_data_file_full_path_by_name(TRAIN_Y_RATINGS)) as f:
		for line in f:
			rating = int(line.split()[0])
			Y.append([rating])

	with open(_get_data_file_full_path_by_name(TRAIN_X_DATES)) as f:
		for line_number, line in enumerate(f):
			date = int(line.split()[0])
			Y[line_number].append(date)

	X = np.array(X, dtype='float')
	Y = np.array(Y, dtype='float')

	assert X.shape == (10000, 2, 99)
	assert Y.shape == (10000, 2)


	X[X == 0] = np.nan
	Y[Y == 0] = np.nan
	
	return X, Y




# "PRIVATE" METHODS

def _get_data_file_full_path_by_name(name):
	return os.path.join(DATA_BASE_DIR, name)
