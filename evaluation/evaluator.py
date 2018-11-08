from tabulate import tabulate
from data_pre_processing.fetch_data import *
import numpy as np
from progressbar import progressbar

# 'regressors' is an array of objects implementing the BaseRegressor Interface
def perform_benchmark(regressors):
	_plot_benchmark_results(_evaluate_regressors(regressors))


# PRIVATE:

def _evaluate_regressors(regressors):
	map_regressor_to_scores = { str(regressor): [] for regressor in regressors }

	for X_train, Y_train, X_test, Y_test in progressbar(training_data_partition_generator()):
		
		for regressor in regressors:
			
			regressor.fit(X_train, Y_train)
			predictions = regressor.predict(X_test)
			score = _rmse(predictions, Y_test[:,1])

			map_regressor_to_scores[str(regressor)].append(score)

	return map_regressor_to_scores


def _plot_benchmark_results(map_regressor_to_scores):
	column_headers = ['Method', 'Standard Deviation', 'Mean']
	rows = [ [regressor_name, np.std(scores), np.mean(scores)]
				for regressor_name, scores in map_regressor_to_scores.items()]

	print(tabulate(rows, headers=column_headers))


def _rmse(predictions, targets):
	return np.sqrt(np.mean((predictions-targets)**2))
	
def training_data_partition_generator(n_runs = 10):
	current_run = 0
	while current_run < n_runs:
		yield random_partition(*get_X_Y_train())
		current_run += 1


