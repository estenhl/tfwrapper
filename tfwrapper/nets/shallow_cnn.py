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

		height, width, channels = X_shape
		fc_input_size = int((height/4) * (width/4) * 64)

		layers = [
			self.reshape([-1, height, width, channels], name=name + '_reshape'),
			self.conv2d(filter=[5, 5], input_depth=channels, depth=32, name=name + '_conv1'),
			self.maxpool2d(k=2, name=name + '_pool1'),
			self.conv2d(filter=[5, 5], input_depth=32, depth=64, name=name + '_conv2'),
			self.conv2d(filter=[5, 5], input_depth=64, depth=64, name=name + '_conv3'),
			self.maxpool2d(k=2, name=name + '_pool2'),
			self.fullyconnected(input_size=fc_input_size, output_size=512, name=name + '_fc'),
			self.dropout(0.8, name=name + '_dropout'),
			self.out([512, self.classes], self.classes, name=name + '_pred')
		]

		super().__init__(X_shape, classes, layers, sess=sess, graph=graph, name=name)