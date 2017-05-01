import numpy as np
import tensorflow as tf

from tfwrapper import TFSession

from .cnn import CNN

class ShallowCNN(CNN):
	learning_rate = 0.001

	def __init__(self, X_shape, classes, sess=None, name='ShallowCNN'):
		if len(X_shape) == 3:
			height, width, self.channels = X_shape
		elif len(X_shape) == 2:
			height, width = X_shape
			self.channels = 1
		self.classes = classes

		height, width, channels = X_shape
		twice_reduce = lambda x: -(-x // 4)
		fc_input_size = twice_reduce(height)*twice_reduce(width)*64

		layers = [
			self.reshape([-1, height, width, channels], name=name + '/reshape'),
			self.conv2d(filter=[5, 5], depth=32, name=name + '/conv1'),
			self.maxpool2d(k=2, name=name + '/pool1'),
			self.conv2d(filter=[5, 5], depth=64, name=name + '/conv2'),
			self.conv2d(filter=[5, 5], depth=64, name=name + '/conv3'),
			self.maxpool2d(k=2, name=name + '/pool2'),
			self.fullyconnected(input_size=fc_input_size, output_size=512, name=name + '/fc'),
			self.dropout(0.8, name=name + '/dropout'),
			self.out([512, self.classes], self.classes, name=name + '/pred')
		]

		with TFSession(sess) as sess:
			super().__init__(X_shape, classes, layers, sess=sess, name=name)