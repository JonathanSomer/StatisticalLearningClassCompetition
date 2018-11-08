# StatisticalLearningClassCompetition

## Useful Links:
* [competition details & instructions](https://www.tau.ac.il/~saharon/StatsLearn2018/Competition18.htm)
* [submission csv](https://docs.google.com/forms/u/2/d/e/1FAIpQLSeG2mUVjFlauDddp-UoEWEVSHwlgY_26ajHinSNzulj0VR0KQ/formResponse)


## How to use this repo:
1. write a regressor which implements the BaseRegressor interface (see `regressors/base_regressor.py`)
2. start the `main.ipynb` notebook.
3. import the regressor class you wrote and add it to the regressors array.
4. run the notebook


## Main Insights:

### Add here any insights you have about the problem.

1. A user can generally tend to rank higher or lower regardless of the specific movie at hand. "Not all 5s were born equal"
2. If a user did not rate a movie we should not simply fill this value with the mean, as we tend not to watch movies we believe we will not enjoy.
