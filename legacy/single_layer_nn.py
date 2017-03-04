import tensorflow as tf
from .nn import NN

class SingleLayerNN(NN):
	def __init__(self, id, input_size, classes, class_weights=None):
		self.input_size = input_size
		super().__init__(id, [input_size], classes)

	def weights(self):
		return {
			'hidden': tf.Variable(tf.random_normal([self.input_size, int(self.input_size * (2/3)) + self.classes]), name='hidden_weight'),
			'out': tf.Variable(tf.random_normal([int(self.input_size * (2/3)) + self.classes, self.classes]), name='out_weight')
		}

	def biases(self):
		return {
			'hidden': tf.Variable(tf.random_normal([int(self.input_size * (2/3)) + self.classes]), name='hidden_bias'),
			'out': tf.Variable(tf.random_normal([self.classes]), name='out_bias')
		}

	def net(self, x, input_shape, weights, biases):
		self.x = tf.placeholder(tf.float32, [None, input_shape[0]], name='x_placeholder')
		self.y = tf.placeholder(tf.float32, [None, self.classes], name='y_placeholder')
		layers = []

		hidden = tf.reshape(self.x, [-1, weights['hidden'].get_shape().as_list()[0]])
		hidden = tf.add(tf.matmul(self.x, weights['hidden']), biases['hidden'])
		hidden = tf.nn.relu(hidden)
		size = str(weights['hidden'].get_shape().as_list()[1])
		layers.append({'name': 'hidden', 'size': size})

		out = tf.add(tf.matmul(hidden, weights['out']), biases['out'], name='out')

		return out, layers