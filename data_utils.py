import numpy as np

DATA_BASE_DIR = './data'
TRAIN_RATINGS = 'train_ratings_all.txt'
# test_dates_all.txt  
# test_ratings_all.txt
# test_y_date.txt
# train_dates_all.txt
# train_ratings_all.txt
# train_y_date.txt
# train_y_rating.txt

# X: for 1000 users, 2 rows - first is the rating of 99 movies, second is the date for each rating
# Y: for 1000 users, rating and date for the "miss congeniality" film
def get_X_Y_train():
	return np.random.rand(1000,2,99), np.random.rand(1000, 2)
