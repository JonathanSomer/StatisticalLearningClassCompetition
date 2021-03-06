import numpy as np
import os.path
from datetime import date

DATA_BASE_DIR = './raw_data'

TEST_X_DATES = 'test_dates_all.txt'
TEST_X_RATINGS = 'test_ratings_all.txt'
TEST_Y_DATES = 'test_y_date.txt'

TRAIN_X_DATES = 'train_dates_all.txt'
TRAIN_X_RATINGS = 'train_ratings_all.txt'
TRAIN_Y_DATES = 'train_y_date.txt'
TRAIN_Y_RATINGS = 'train_y_rating.txt'

MOVIE_DATES = 'movie_titles.txt'

NUMBER_OF_MOVIES = 99
NUMBER_OF_USERS_TRAIN = 10000
NUMBER_OF_USERS_TEST = 2931

OUTPUT_MOVIE_YEAR = 2000



# X: for 10000 users, 3 rows:
# first: date of 99 movies, 
# second: rating for each movie, 
# third: date of target movie prediciton
# Y: for 10000 users, (date, rating) for the "miss congeniality" film
def get_X_Y_train():
    X = _get_X(train=True)
    Y = _get_Y()

    assert X.shape == (NUMBER_OF_USERS_TRAIN, 3, NUMBER_OF_MOVIES)
    assert Y.shape == (NUMBER_OF_USERS_TRAIN,)

    return X, Y


def get_X_test():
    X = _get_X(train=False)
    assert X.shape == (NUMBER_OF_USERS_TEST, 3, NUMBER_OF_MOVIES)

    return X


def get_movies_dates():
    with open(_get_data_file_full_path_by_name(MOVIE_DATES)) as f:
        lines = f.readlines()
    lines = [_transform_year_to_days_since_epoch(int(x.split(',')[0].strip()))
             for x in lines[:NUMBER_OF_MOVIES]]
    return np.array(lines, dtype='float')


def get_output_movie_date():
    # return _transform_year_to_days_since_epoch(OUTPUT_MOVIE_YEAR)
    X_train, _ = get_X_Y_train()
    return np.min(X_train[:,2,0])

def random_partition(X, Y, training_fraction = 0.9):
    train_size = int(training_fraction*len(X))
    test_size = len(X) - train_size

    mask = np.array([True]*train_size + [False]*test_size)
    np.random.shuffle(mask)

    X_train = X[mask]
    Y_train = Y[mask]
    X_validation = X[~mask]
    Y_validation = Y[~mask]

    return X_train, Y_train, X_validation, Y_validation


def random_partition_one(X, Y):
    train_size = len(X) - 1
    test_size = 1

    mask = np.array([True]*train_size + [False]*test_size)
    np.random.shuffle(mask)

    X_train = X[mask]
    Y_train = Y[mask]
    X_validation = X[~mask]
    Y_validation = Y[~mask]

    return X_train, Y_train, X_validation, Y_validation


# "PRIVATE" METHODS

def _get_data_file_full_path_by_name(name):
    return os.path.join(DATA_BASE_DIR, name)


def _transform_year_to_days_since_epoch(year):
    d0 = date(year, 7, 1)
    d1 = date(1997, 1, 1)
    delta = d0 - d1
    return float(delta.days)


def _get_X(train=True):
    X = []

    dates_file_name = TRAIN_X_DATES if train else TEST_X_DATES
    ratings_file_name = TRAIN_X_RATINGS if train else TEST_X_RATINGS
    target_movie_dates_file_name = TRAIN_Y_DATES if train else TEST_Y_DATES

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

    with open(_get_data_file_full_path_by_name(target_movie_dates_file_name)) as f:
        for line_number, line in enumerate(f):
            date = int(line.split()[0])
            X[line_number].append(np.repeat(date, NUMBER_OF_MOVIES))
            # X[line_number].append([date])

    X = np.array(X, dtype='float')
    X[X == 0] = np.nan

    return X


def _get_Y():
    Y = []

    with open(_get_data_file_full_path_by_name(TRAIN_Y_RATINGS)) as f:
        for line_number, line in enumerate(f):
            rating = int(line.split()[0])
            Y.append(rating)

    Y = np.array(Y, dtype='float')
    Y[Y == 0] = np.nan
    return Y
