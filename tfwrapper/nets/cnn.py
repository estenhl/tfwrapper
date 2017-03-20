import tensorflow as tf

from .neural_net import NeuralNet

class CNN(NeuralNet):
	def __init__(self, X_shape, y_size, layers, sess=None, graph=None, name='NeuralNet'):
		super().__init__(X_shape, y_size, layers, sess=sess, graph=graph, name=name)

	def conv2d(self, weight_shape, bias_size, strides=1, padding='SAME', name='conv2d'):
		weight_name = name + '_W'
		bias_name = name + '_b'

		def create_layer(x):
			weight = tf.Variable(tf.random_normal(weight_shape), name=weight_name)
			bias = tf.Variable(tf.random_normal([bias_size]), name=bias_name)
			conv = tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding=padding, name=name)
			conv = tf.nn.bias_add(conv, bias)

			return tf.nn.relu(conv)

		return create_layer

	def maxpool2d(self, k=2, padding='SAME', name='maxpool2d'):
		return lambda x: tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding, name=name)