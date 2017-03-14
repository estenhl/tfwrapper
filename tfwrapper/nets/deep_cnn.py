import tensorflow as tf

from .cnn import CNN

class DeepCNN(CNN):
	def __init__(self, X_shape, classes, sess=None, graph=None, name='NeuralNet'):
		if len(X_shape) == 3:
			height, width, self.channels = X_shape
		elif len(X_shape) == 2:
			height, width = X_shape
			self.channels = 1
		self.classes = classes
		
		layers = self.layers(X_shape, name)

		super().__init__(X_shape, classes, layers, sess=sess, graph=graph, name=name)

	def layers(self, input_shape, name):
		height, width, channels = input_shape

		weights = {
			'conv1': [5, 5, self.channels, 32],
			'conv2': [5, 5, 32, 64],
			'conv3': [5, 5, 64, 64],
			'conv4': [5, 5, 64, 128],
			'conv5': [5, 5, 128, 128],
			'conv6': [5, 5, 128, 256],
			'conv7': [3, 3, 256, 256],
			'conv8': [3, 3, 256, 512],
			'fc1': [512, 1024],
			'fc2': [1024, 512],
			'out': [512, self.classes]
		}

		biases = {
			'conv1': 32,
			'conv2': 64,
			'conv3': 64,
			'conv4': 128,
			'conv5': 128,
			'conv6': 256,
			'conv7': 256,
			'conv8': 512,
			'fc1': 1024,
			'fc2': 512,
			'out': self.classes
		}

		k1 = 2
		k2 = 2
		k3 = 2
		k4 = height / (k1*k2*k3)
		
		layers = [
			lambda x: tf.reshape(x, shape=[-1, height, width, channels]),
			lambda x: self.conv2d(x, self.weight(weights['conv1'], name=self.name + '_wc1'), self.bias(biases['conv1'], name=self.name + '_bc1'), name=name + '_conv1'),
			lambda x: self.maxpool2d(x, k=k1, name=name + '_pool1'),
			lambda x: self.conv2d(x, self.weight(weights['conv2'], name=self.name + '_wc2'), self.bias(biases['conv2'], name=self.name + '_bc2'), name=name + '_conv2'),
			lambda x: self.maxpool2d(x, k=k2, name=name + '_pool2'),
			lambda x: self.conv2d(x, self.weight(weights['conv3'], name=self.name + '_wc3'), self.bias(biases['conv3'], name=self.name + '_bc3'), name=name + '_conv3'),
			lambda x: self.conv2d(x, self.weight(weights['conv4'], name=self.name + '_wc4'), self.bias(biases['conv4'], name=self.name + '_bc4'), name=name + '_conv4'),
			lambda x: self.maxpool2d(x, k=k3, name=name + '_pool3'),
			lambda x: self.conv2d(x, self.weight(weights['conv5'], name=self.name + '_wc5'), self.bias(biases['conv5'], name=self.name + '_bc5'), name=name + '_conv5'),
			lambda x: self.conv2d(x, self.weight(weights['conv6'], name=self.name + '_wc6'), self.bias(biases['conv6'], name=self.name + '_bc6'), name=name + '_conv6'),
			lambda x: self.conv2d(x, self.weight(weights['conv7'], name=self.name + '_wc7'), self.bias(biases['conv7'], name=self.name + '_bc7'), name=name + '_conv7'),
			lambda x: self.conv2d(x, self.weight(weights['conv8'], name=self.name + '_wc8'), self.bias(biases['conv8'], name=self.name + '_bc8'), name=name + '_conv8'),
			lambda x: self.maxpool2d(x, k=k4, name=name + '_flatten'),
			lambda x: self.fullyconnected(x, self.weight(weights['fc1'], name=self.name + '_fc1'), self.bias(biases['fc1'], name=self.name + '_fc1'), name=name + '_fc1'),
			lambda x: self.fullyconnected(x, self.weight(weights['fc2'], name=self.name + '_fc2'), self.bias(biases['fc2'], name=self.name + '_fc2'), name=name + '_fc2'),
			lambda x: tf.add(tf.matmul(x, self.weight(weights['out'], name=self.name + '_wout')), self.bias(biases['out'], name=self.name + '_bout'), name=name + '_pred')
		]
		
		return layers