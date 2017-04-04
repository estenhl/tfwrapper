import numpy as np
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.nets import NeuralNet

class SingleLayerNeuralNet(NeuralNet):
	def __init__(self, X_shape, y_size, hidden, sess=None, name='SingleLayerNeuralNet'):
		with TFSession(sess) as sess:
			X_size = np.prod(X_shape)

			layers = [
				self.fullyconnected(input_size=X_size, output_size=hidden, name=name + '_hidden'),
				self.out([hidden, y_size], y_size, name=name + '_pred')
			]

			super().__init__(X_shape, y_size, layers, sess=sess, name=name)
			sess.run(tf.global_variables_initializer())

			self.graph = sess.graph
