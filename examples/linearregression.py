from tfwrapper.regression import LinearRegression
from tfwrapper.datasets import wine

dataset = wine(y='Alcohol')
reg = LinearRegression(dataset.X.shape[1], 1, len(dataset.X))
reg.train(dataset.X, dataset.y, epochs=10000)