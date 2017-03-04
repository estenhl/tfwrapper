import tensorflow as tf

from .cnn import CNN

class DeepCNN(CNN):
	def __init__(self, X_shape, classes, sess=None, name='NeuralNet'):
		if len(X_shape) == 3:
			height, width, channels = X_shape
		elif len(X_shape) == 2:
			height, width = X_shape
			channels = 1
		
		weights = self.weights(channels, classes, name)
		biases = self.biases(classes, name)
		layers = self.layers(X_shape, weights, biases, name)

		super().__init__(X_shape, classes, layers, sess=sess, name=name)

	def weights(self, channels, classes, name):
		return {
			'wc1': tf.Variable(tf.random_normal([5, 5, channels, 32]), name=name + '_wc1'),
			'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name=name + '_wc2'),
			'wc3': tf.Variable(tf.random_normal([5, 5, 64, 64]), name=name + '_wc3'),
			'wc4': tf.Variable(tf.random_normal([5, 5, 64, 128]), name=name + '_wc4'),
			'wc5': tf.Variable(tf.random_normal([5, 5, 128, 128]), name=name + '_wc5'),
			'wc6': tf.Variable(tf.random_normal([5, 5, 128, 256]), name=name + '_wc6'),
			'wc7': tf.Variable(tf.random_normal([3, 3, 256, 256]), name=name + '_wc7'),
			'wc8': tf.Variable(tf.random_normal([3, 3, 256, 512]), name=name + '_wc8'),
			'wd1': tf.Variable(tf.random_normal([512, 1024]), name=name + '_wd1'),
			'wd2': tf.Variable(tf.random_normal([1024, 512]), name=name + '_wd2'),
			'out': tf.Variable(tf.random_normal([512, classes]), name=name + '_out_weight')
		}

	def biases(self, classes, name):
		return {
			'bc1': tf.Variable(tf.random_normal([32]), name=name + '_bc1'),
			'bc2': tf.Variable(tf.random_normal([64]), name=name + '_bc2'),
			'bc3': tf.Variable(tf.random_normal([64]), name=name + '_bc3'),
			'bc4': tf.Variable(tf.random_normal([128]), name=name + '_bc4'),
			'bc5': tf.Variable(tf.random_normal([128]), name=name + '_bc5'),
			'bc6': tf.Variable(tf.random_normal([256]), name=name + '_bc6'),
			'bc7': tf.Variable(tf.random_normal([256]), name=name + '_bc7'),
			'bc8': tf.Variable(tf.random_normal([512]), name=name + '_bc8'),
			'bd1': tf.Variable(tf.random_normal([1024]), name=name + '_bd1'),
			'bd2': tf.Variable(tf.random_normal([512]), name=name + '_bd2'),
			'out': tf.Variable(tf.random_normal([classes]), name=name + '_out_bias')
		}

	def layers(self, input_shape, weights, biases, name):
		height, width, channels = input_shape

		k1 = 2
		k2 = 2
		k3 = 2
		k4 = height / (k1*k2*k3)
		
		layers = [
			lambda x: tf.reshape(x, shape=[-1, height, width, channels]),
			lambda x: self.conv2d(x, weights['wc1'], biases['bc1'], name=name + '_conv1'),
			lambda x: self.maxpool2d(x, k=k1, name=name + '_pool1'),
			lambda x: self.conv2d(x, weights['wc2'], biases['bc2'], name=name + '_conv2'),
			lambda x: self.maxpool2d(x, k=k2, name=name + '_pool2'),
			lambda x: self.conv2d(x, weights['wc3'], biases['bc3'], name=name + '_conv3'),
			lambda x: self.conv2d(x, weights['wc4'], biases['bc4'], name=name + '_conv4'),
			lambda x: self.maxpool2d(x, k=k3, name=name + '_pool3'),
			lambda x: self.conv2d(x, weights['wc5'], biases['bc4'], name=name + '_conv5'),
			lambda x: self.conv2d(x, weights['wc6'], biases['bc6'], name=name + '_conv6'),
			lambda x: self.conv2d(x, weights['wc7'], biases['bc7'], name=name + '_conv7'),
			lambda x: self.conv2d(x, weights['wc8'], biases['bc8'], name=name + '_conv8'),
			lambda x: self.maxpool2d(x, k=k4, name=name + '_flatten'),
			lambda x: self.fullyconnected(x, weights['wd1'], biases['bd1'], name=name + '_fc1'),
			lambda x: self.fullyconnected(x, weights['wd2'], biases['bd2'], name=name + '_fc2'),
			lambda x: tf.add(tf.matmul(x, weights['out']), biases['out'], name=name + '_out')
		]
		
		return layers