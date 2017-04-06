import numpy as np
import tensorflow as tf

from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.utils.exceptions import InvalidArgumentException

def test_mismatching_lengths():
	model = SingleLayerNeuralNet([28, 28, 1], 3, 5)
	X = np.zeros([50, 28, 28, 1])
	y = np.zeros([100, 3])

	exception = False
	try:
		model.train(X, y)
	except InvalidArgumentException:
		exception = True

	assert exception

def test_invalid_X_shape():
	model = SingleLayerNeuralNet([28, 28, 1], 3, 5)
	X = np.zeros([100, 28, 28, 2])
	y = np.zeros([100, 3])

	exception = False
	try:
		model.train(X, y)
	except InvalidArgumentException:
		exception = True

	assert exception

def test_y_without_onehot():
	model = SingleLayerNeuralNet([28, 28, 1], 3, 5)
	X = np.zeros([100, 28, 28, 1])
	y = np.zeros([100])

	exception = False
	try:
		model.train(X, y)
	except InvalidArgumentException:
		exception = True

	assert exception

def test_invalid_classes():
	model = SingleLayerNeuralNet([28, 28, 1], 3, 5)
	X = np.zeros([100, 28, 28, 1])
	y = np.zeros([100, 5])

	exception = False
	try:
		model.train(X, y)
	except InvalidArgumentException:
		exception = True

	assert exception