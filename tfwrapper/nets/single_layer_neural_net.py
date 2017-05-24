import numpy as np
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.nets import NeuralNet
from tfwrapper.layers import fullyconnected, out

class SingleLayerNeuralNet(NeuralNet):
	def __init__(self, X_shape, y_size, hidden, sess=None, name='SingleLayerNeuralNet'):
		with TFSession(sess) as sess:
			X_size = np.prod(X_shape)

			layers = [
				fullyconnected(inputs=X_size, outputs=hidden, name=name + '/hidden'),
				out(inputs=hidden, outputs=y_size, name=name + '/pred')
			]

			super().__init__(X_shape, y_size, layers, sess=sess, name=name)

			self.graph = sess.graph
