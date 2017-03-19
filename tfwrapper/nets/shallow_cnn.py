import numpy as np
import tensorflow as tf

from .cnn import CNN

class ShallowCNN(CNN):
	learning_rate = 0.001

	def __init__(self, X_shape, classes, sess=None, graph=None, name='ShallowCNN'):

		if len(X_shape) == 3:
			height, width, self.channels = X_shape
		elif len(X_shape) == 2:
			height, width = X_shape
			self.channels = 1
		self.classes = classes

		if not (height % 4 == 0 and width % 4 == 0):
			raise Exception('Height and width must be multiples of 4!')
		
		layers = self.layers(X_shape, name)

		super().__init__(X_shape, classes, layers, sess=sess, graph=graph, name=name)

	def layers(self, X_shape, name):
		height, width, channels = X_shape
		fc_input_size = int((height/4) * (width/4) * 64)

		weights = {
			'conv1': [5, 5, channels, 32],
			'conv2': [5, 5, 32, 64],
			'conv3': [5, 5, 64, 64],
			'fc': [fc_input_size, 512],
			'out': [512, self.classes]
		}

		biases = {
			'conv1': 32,
			'conv2': 64,
			'conv3': 64,
			'fc': 512,
			'out': self.classes
		}
		
		layers = [
			lambda x: tf.reshape(x, shape=[-1, height, width, channels]),
			lambda x: self.conv2d(x, self.weight(weights['conv1'], name=name + '_wc1'), self.bias(biases['conv1'], name=name + '_bc1'), name=name + '_conv1'),
			lambda x: self.maxpool2d(x, k=2, name=name + '_pool1'),
			lambda x: self.conv2d(x, self.weight(weights['conv2'], name=name + '_wc2'), self.bias(biases['conv2'], name=name + '_bc2'), name=name + '_conv2'),
			lambda x: self.conv2d(x, self.weight(weights['conv3'], name=name + '_wc3'), self.bias(biases['conv3'], name=name + '_bc3'), name=name + '_conv3'),
			lambda x: self.maxpool2d(x, k=2, name=name + '_pool2'),
			lambda x: self.fullyconnected(x, self.weight(weights['fc'], name=name + '_fc'), self.bias(biases['fc'], name=name + '_fc'), name=name + '_fc1'),
			lambda x: tf.nn.dropout(x, 0.8, name=name + '_dropout'),
			lambda x: tf.add(tf.matmul(x, self.weight(weights['out'], name=name + '_wout')), self.bias(biases['out'], name=name + '_bout'), name=name + '_pred')
		]
		
		return layers
