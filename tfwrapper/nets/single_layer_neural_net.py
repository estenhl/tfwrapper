import numpy as np
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.nets import NeuralNet

class SingleLayerNeuralNet(NeuralNet):
	def __init__(self, X_shape, y_size, hidden, sess=None, graph=None, name='SingleLayerNeuralNet'):
		if graph is None:
			if sess is not None:
				raise Exception('When a session is passed, a graph must be passed aswell')
			graph = tf.Graph()

		with graph.as_default():
			with TFSession(sess, graph) as sess:
				X_size = np.prod(X_shape)

				layers = [
					self.fullyconnected([X_size, hidden], hidden, name=name + '_hidden'),
					self.out([hidden, y_size], y_size, name=name + '_pred')
				]

				super().__init__(X_shape, y_size, layers, sess=sess, graph=graph, name=name)
				sess.run(tf.global_variables_initializer())

		self.graph = graph
