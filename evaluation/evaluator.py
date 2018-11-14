from tabulate import tabulate
from data_pre_processing.fetch_data import *
import numpy as np
import sys
from IPython.core.display import clear_output

N_RUNS = 50

# 'regressors' is an array of objects implementing the BaseRegressor Interface
def perform_benchmark(regressors, n_runs = N_RUNS):
	_plot_benchmark_results(_evaluate_regressors(regressors, n_runs))


# PRIVATE:

def _evaluate_regressors(regressors, n_runs):
	map_regressor_to_scores = { str(regressor): [] for regressor in regressors }

	for X_train, Y_train, X_test, Y_test in training_data_partition_generator(n_runs):

		for regressor in regressors:
			
			_X_train, _Y_train, _X_test, _Y_test = np.copy(X_train), np.copy(Y_train), np.copy(X_test), np.copy(Y_test)
			
			regressor.fit(_X_train, _Y_train)
			predictions = regressor.predict(_X_test)
			score = _rmse(predictions, _Y_test)

			map_regressor_to_scores[str(regressor)].append(score)
		_plot_benchmark_results(map_regressor_to_scores)

	return map_regressor_to_scores


def _plot_benchmark_results(map_regressor_to_scores):
	clear_output()
	column_headers = ['Method', 'Standard Deviation', 'Mean']
	rows = [ [regressor_name, np.std(scores), np.mean(scores)]
				for regressor_name, scores in map_regressor_to_scores.items()]

	print("Run #{}".format(len(next(iter(map_regressor_to_scores.values())))), sep=' ', end='\n', flush=True)
	print(tabulate(rows, headers=column_headers), sep=' ', end='', flush=True)


def _rmse(predictions, targets):
	return np.sqrt(np.mean((predictions-targets)**2))
	
def training_data_partition_generator(n_runs):
	current_run = 0
	for i in range(n_runs):		
		yield random_partition(*get_X_Y_train())


