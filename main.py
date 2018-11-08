from data_pre_processing.fetch_data import *
from regressors import *


X_train, Y_train = get_X_Y_train()
X_test, Y_test = get_X_Y_test()

X_train, Y_train, X_validation, Y_validation = random_partition(X_train, Y_train)

# import pdb; pdb.set_trace()
