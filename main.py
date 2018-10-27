from data_utils import *

X_train, Y_train = get_X_Y_train()
X_test, Y_test = get_X_Y_test()

X_train, Y_train, X_validation, Y_validation = random_partition(X_train, Y_train)
