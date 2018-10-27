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
NUMBER_OF_USERS_TRAIN = 10000
NUMBER_OF_USERS_TEST = 2931

# X: for 1000 users, 2 rows - first is the date of 99 movies, second is the rating for each rating
# Y: for 1000 users, (date, rating) for the "miss congeniality" film
def get_X_Y_train():
	Y = _get_Y(train=True)
	X = _get_X(train=True)

	assert X.shape == (NUMBER_OF_USERS_TRAIN, 2, NUMBER_OF_MOVIES)
	assert Y.shape == (NUMBER_OF_USERS_TRAIN, 2)

	return X, Y

def get_X_Y_test():
	Y = _get_Y(train=False)
	X = _get_X(train=False)

	assert X.shape == (NUMBER_OF_USERS_TEST, 2, NUMBER_OF_MOVIES)
	assert Y.shape == (NUMBER_OF_USERS_TEST, 1)

	return X, Y

# "PRIVATE" METHODS

def _get_data_file_full_path_by_name(name):
	return os.path.join(DATA_BASE_DIR, name)

def _get_X(train=True):
	X = []

	dates_file_name = TRAIN_X_DATES if train else TEST_X_DATES
	ratings_file_name = TRAIN_X_RATINGS if train else TEST_X_RATINGS

	with open(_get_data_file_full_path_by_name(dates_file_name)) as f:
		for line in f:
			dates = [int(date) for date in line.split()]
			assert len(dates) == NUMBER_OF_MOVIES
			X.append([dates])

	with open(_get_data_file_full_path_by_name(ratings_file_name)) as f:
		for line_number, line in enumerate(f):
			ratings = [int(rating) for rating in line.split()]
			assert len(ratings) == NUMBER_OF_MOVIES
			X[line_number].append(ratings)

	X = np.array(X, dtype='float')
	X[X == 0] = np.nan

	return X

def _get_Y(train=True):
	Y = []

	dates_file_name = TRAIN_Y_DATES if train else TEST_Y_DATES

	with open(_get_data_file_full_path_by_name(dates_file_name)) as f:
		for line in f:
			date = int(line.split()[0])
			Y.append([date])

	if train:
		with open(_get_data_file_full_path_by_name(TRAIN_Y_RATINGS)) as f:
			for line_number, line in enumerate(f):
				rating = int(line.split()[0])
				Y[line_number].append(rating)

	Y = np.array(Y, dtype='float')
	Y[Y == 0] = np.nan
	return Y
