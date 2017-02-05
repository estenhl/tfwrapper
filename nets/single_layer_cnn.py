import tensorflow as tf
from .cnn import CNN

class SingleLayerCNN(CNN):
	def __init__(self, id, input_shape, classes, class_weights=None):
		super().__init__(id, input_shape, classes, class_weights)

	def weights(self):
		height, width, channels = self.input_shape
		return {
			'wc1': tf.Variable(tf.random_normal([3, 3, channels, channels]), name='wc1'),
			'wd1': tf.Variable(tf.random_normal([channels, channels * 4]), name='wd1'),
			'out': tf.Variable(tf.random_normal([channels * 4, self.classes]), name='out_weight')
		}

	def biases(self):
		height, width, channels = self.input_shape
		return {
			'bc1': tf.Variable(tf.random_normal([channels]), name='bc1'),
			'bd1': tf.Variable(tf.random_normal([channels * 4]), name='bd1'),
			'out': tf.Variable(tf.random_normal([self.classes]), name='out_bias')
		}

	def net(self, x, input_shape, weights, biases):
		height, width, channels = input_shape
		x = tf.reshape(x, shape=[-1, height, width, channels])
		layers = []

		# Conv1
		conv = self.conv2d(x, weights['wc1'], biases['bc1'], padding='VALID', name='conv1')
		depth = weights['wc1'].get_shape().as_list()[3]
		size = str(input_shape[0]) + 'x' + str(input_shape[1]) + 'x' + str(depth)
		layers.append({'name': 'conv', 'size': size})

		# Fully connected
		fc = tf.reshape(conv, [-1, weights['wd1'].get_shape().as_list()[0]])
		fc = tf.add(tf.matmul(fc, weights['wd1']), biases['bd1'])
		fc = tf.nn.relu(fc)
		size = str(weights['wd1'].get_shape().as_list()[1])
		layers.append({'name': 'fc1', 'size': size})

		# Output
		out = tf.add(tf.matmul(fc, weights['out']), biases['out'], name='out')
		size = str(weights['out'].get_shape().as_list()[1])
		layers.append({'layer': out, 'name': 'out', 'size': size})

		return out, layers