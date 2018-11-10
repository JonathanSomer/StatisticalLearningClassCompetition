import numpy as np
import pandas as pd


def fill_with_mean(X):
    assert len(X.shape) == 3 and X.shape[1] == 2
    X = X.reshape(X.shape[0], -1)
    X = pd.DataFrame(X)
    X = X.fillna(X.mean())
    X = np.array(X)
    return X.reshape(X.shape[0], 2, -1)


def fill_with_zeros(X):
    return np.nan_to_num(X)


def fill_with_num(X, num):
    filled_matrix = fill_with_zeros(X)
    filled_matrix[filled_matrix == 0] = num
    return filled_matrix

# coefficient found over cross validation
def fill_ratings_with_mean_per_user(X, coefficient = 1.7):
    _X = np.copy(X)
    
    all_users_ratings = _X[:,1,:]
    
    new_ratings = []
    mean_per_user = []

    for user_ratings in all_users_ratings:
        ratings_no_nan = user_ratings[~np.isnan(user_ratings)]
        user_mean = np.mean(ratings_no_nan)
        user_std = np.std(ratings_no_nan)

        user_ratings[np.isnan(user_ratings)] = user_mean - coefficient*user_std
        
        new_ratings.append(user_ratings)
        mean_per_user.append(user_mean)


    new_X = np.copy(X)
    new_X[:,1,:] = np.array(new_ratings)

    return new_X, mean_per_user
