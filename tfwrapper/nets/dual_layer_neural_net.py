import numpy as np
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.nets import NeuralNet

class DualLayerNeuralNet(NeuralNet):
	def __init__(self, X_shape, y_size, hidden1, hidden2, sess=None, name='DualLayerNeuralNet'):
		with TFSession(sess) as sess:
			X_size = np.prod(X_shape)

			layers = [
				self.fullyconnected(inputs=X_size, outputs=hidden1, name=name + '/hidden1'),
				self.fullyconnected(inputs=hidden1, outputs=hidden2, name=name + '/hidden2'),
				self.out(inputs=hidden2, outputs=y_size, name=name + '/pred')
			]

			super().__init__(X_shape, y_size, layers, sess=sess, graph=graph, name=name)

			self.graph = sess.graph