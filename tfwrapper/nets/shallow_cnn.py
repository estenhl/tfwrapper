import numpy as np
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.layers import reshape, conv2d, maxpool2d, fullyconnected, dropout, out

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
			reshape([-1, height, width, channels], name=name + '/reshape'),
			conv2d(filter=[5, 5], depth=32, name=name + '/conv1'),
			maxpool2d(k=2, name=name + '/pool1'),
			conv2d(filter=[5, 5], depth=64, name=name + '/conv2'),
			conv2d(filter=[5, 5], depth=64, name=name + '/conv3'),
			maxpool2d(k=2, name=name + '/pool2'),
			fullyconnected(inputs=fc_input_size, outputs=512, name=name + '/fc'),
			dropout(0.8, name=name + '/dropout'),
			out(inputs=512, outputs=classes, name=name + '/pred')
		]

		with TFSession(sess) as sess:
			super().__init__(X_shape, classes, layers, sess=sess, name=name)