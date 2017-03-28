import numpy as np
import tensorflow as tf

from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.utils.exceptions import InvalidArgumentException

def test_mismatching_lengths():
	with tf.Session() as sess:
		model = SingleLayerNeuralNet([28, 28, 1], 3, 5, sess=sess, graph=sess.graph)
		X = np.zeros([50, 28, 28, 1])
		y = np.zeros([100, 3])

		exception = False
		try:
			model.train(X, y, sess=sess)
		except InvalidArgumentException:
			exception = True

		assert exception

def test_invalid_X_shape():
	with tf.Session() as sess:
		model = SingleLayerNeuralNet([28, 28, 1], 3, 5, sess=sess, graph=sess.graph)
		X = np.zeros([100, 28, 28, 2])
		y = np.zeros([100, 3])

		exception = False
		try:
			model.train(X, y, sess=sess)
		except InvalidArgumentException:
			exception = True

		assert exception

def test_y_without_onehot():
	with tf.Session() as sess:
		model = SingleLayerNeuralNet([28, 28, 1], 3, 5, sess=sess, graph=sess.graph)
		X = np.zeros([100, 28, 28, 1])
		y = np.zeros([100])

		exception = False
		try:
			model.train(X, y, sess=sess)
		except InvalidArgumentException:
			exception = True

		assert exception

def test_invalid_classes():
	with tf.Session() as sess:
		model = SingleLayerNeuralNet([28, 28, 1], 3, 5, sess=sess, graph=sess.graph)
		X = np.zeros([100, 28, 28, 1])
		y = np.zeros([100, 5])

		exception = False
		try:
			model.train(X, y, sess=sess)
		except InvalidArgumentException:
			exception = True

		assert exception