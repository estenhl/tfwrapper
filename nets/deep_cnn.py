import tensorflow as tf

from .cnn import CNN

class DeepCNN(CNN):
	def __init__(self, X_shape, classes, sess=None, graph=None, name='NeuralNet'):
		if len(X_shape) == 3:
			height, width, self.channels = X_shape
		elif len(X_shape) == 2:
			height, width = X_shape
			channels = 1
		self.classes = classes
		
		layers = self.layers(X_shape, name)

		super().__init__(X_shape, classes, layers, sess=sess, graph=graph, name=name)

	def weights(self):
		return {
			'wc1': tf.Variable(tf.random_normal([5, 5, self.channels, 32]), name=self.name + '_wc1'),
			'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name=self.name + '_wc2'),
			'wc3': tf.Variable(tf.random_normal([5, 5, 64, 64]), name=self.name + '_wc3'),
			'wc4': tf.Variable(tf.random_normal([5, 5, 64, 128]), name=self.name + '_wc4'),
			'wc5': tf.Variable(tf.random_normal([5, 5, 128, 128]), name=self.name + '_wc5'),
			'wc6': tf.Variable(tf.random_normal([5, 5, 128, 256]), name=self.name + '_wc6'),
			'wc7': tf.Variable(tf.random_normal([3, 3, 256, 256]), name=self.name + '_wc7'),
			'wc8': tf.Variable(tf.random_normal([3, 3, 256, 512]), name=self.name + '_wc8'),
			'wd1': tf.Variable(tf.random_normal([512, 1024]), name=self.name + '_wd1'),
			'wd2': tf.Variable(tf.random_normal([1024, 512]), name=self.name + '_wd2'),
			'out': tf.Variable(tf.random_normal([512, self.classes]), name=self.name + '_out_weight')
		}

	def biases(self):
		return {
			'bc1': tf.Variable(tf.random_normal([32]), name=self.name + '_bc1'),
			'bc2': tf.Variable(tf.random_normal([64]), name=self.name + '_bc2'),
			'bc3': tf.Variable(tf.random_normal([64]), name=self.name + '_bc3'),
			'bc4': tf.Variable(tf.random_normal([128]), name=self.name + '_bc4'),
			'bc5': tf.Variable(tf.random_normal([128]), name=self.name + '_bc5'),
			'bc6': tf.Variable(tf.random_normal([256]), name=self.name + '_bc6'),
			'bc7': tf.Variable(tf.random_normal([256]), name=self.name + '_bc7'),
			'bc8': tf.Variable(tf.random_normal([512]), name=self.name + '_bc8'),
			'bd1': tf.Variable(tf.random_normal([1024]), name=self.name + '_bd1'),
			'bd2': tf.Variable(tf.random_normal([512]), name=self.name + '_bd2'),
			'out': tf.Variable(tf.random_normal([self.classes]), name=self.name + '_out_bias')
		}

	def layers(self, input_shape, name):
		height, width, channels = input_shape

		k1 = 2
		k2 = 2
		k3 = 2
		k4 = height / (k1*k2*k3)
		
		layers = [
			lambda x: tf.reshape(x, shape=[-1, height, width, channels]),
			lambda x: self.conv2d(x, self.weights()['wc1'], self.biases()['bc1'], name=name + '_conv1'),
			lambda x: self.maxpool2d(x, k=k1, name=name + '_pool1'),
			lambda x: self.conv2d(x, self.weights()['wc2'], self.biases()['bc2'], name=name + '_conv2'),
			lambda x: self.maxpool2d(x, k=k2, name=name + '_pool2'),
			lambda x: self.conv2d(x, self.weights()['wc3'], self.biases()['bc3'], name=name + '_conv3'),
			lambda x: self.conv2d(x, self.weights()['wc4'], self.biases()['bc4'], name=name + '_conv4'),
			lambda x: self.maxpool2d(x, k=k3, name=name + '_pool3'),
			lambda x: self.conv2d(x, self.weights()['wc5'], self.biases()['bc4'], name=name + '_conv5'),
			lambda x: self.conv2d(x, self.weights()['wc6'], self.biases()['bc6'], name=name + '_conv6'),
			lambda x: self.conv2d(x, self.weights()['wc7'], self.biases()['bc7'], name=name + '_conv7'),
			lambda x: self.conv2d(x, self.weights()['wc8'], self.biases()['bc8'], name=name + '_conv8'),
			lambda x: self.maxpool2d(x, k=k4, name=name + '_flatten'),
			lambda x: self.fullyconnected(x, self.weights()['wd1'], self.biases()['bd1'], name=name + '_fc1'),
			lambda x: self.fullyconnected(x, self.weights()['wd2'], self.biases()['bd2'], name=name + '_fc2'),
			lambda x: tf.add(tf.matmul(x, self.weights()['out']), self.biases()['out'], name=name + '_out')
		]
		
		return layers