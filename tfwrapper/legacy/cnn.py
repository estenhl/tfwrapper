import tensorflow as tf
from .nn import NN

class CNN(NN):
	def __init__(self, id, input_shape, classes, class_weights=None):
		super().__init__(id, input_shape, classes, class_weights=class_weights)

	def conv2d(self, x, W, b, strides=1, name=None, padding='SAME'):
		conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding, name=name)
		conv = tf.nn.bias_add(conv, b)

		return tf.nn.relu(conv)

	def maxpool2d(self, x, k=2, name=None):
		return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')