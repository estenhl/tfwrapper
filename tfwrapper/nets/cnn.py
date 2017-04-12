import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.utils.exceptions import InvalidArgumentException

from .neural_net import NeuralNet

class CNN(NeuralNet):
	def __init__(self, X_shape, classes, layers, sess=None, name='NeuralNet'):
		with TFSession(sess) as sess:
			super().__init__(X_shape, classes, layers, sess=sess, name=name)

	@staticmethod
	def conv2d(*, filter, input_depth, depth, strides=1, padding='SAME', trainable=True, name='conv2d'):
		if len(filter) != 2:
			raise InvalidArgumentException('conv2d takes filters with exactly 2 dimensions (e.g. [3, 3])')

		weight_shape = filter + [input_depth, depth]
		bias_size = depth
		weight_name = name + '/W'
		bias_name = name + '/b'

		def create_layer(x):
			weight = CNN.weight(weight_shape, name=weight_name, trainable=trainable)
			bias = CNN.bias(bias_size, name=bias_name, trainable=trainable)
			conv = tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding=padding, name=name)
			conv = tf.nn.bias_add(conv, bias)

			return tf.nn.relu(conv)

		return create_layer

	@staticmethod
	def maxpool2d(*, k=2, padding='SAME', name='maxpool2d'):
		return lambda x: tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding, name=name)