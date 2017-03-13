import tensorflow as tf

from .neural_net import NeuralNet

class CNN(NeuralNet):
	def __init__(self, X_shape, y_size, layers, sess=None, graph=None, name='NeuralNet'):
		super().__init__(X_shape, y_size, layers, sess=sess, graph=graph, name=name)

	def conv2d(self, x, W, b, strides=1, padding='SAME', name=None):
		conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding, name=name)
		conv = tf.nn.bias_add(conv, b)

		return tf.nn.relu(conv)

	def maxpool2d(self, x, k=2, padding='SAME', name=None):
		return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)