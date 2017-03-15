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
					lambda x: self.fullyconnected(x, self.weight([X_size, hidden], name=name + '_hidden_weight'), self.bias(hidden, name=name + '_hidden_bias'), name=name + '_hidden'),
					lambda x: tf.add(tf.matmul(x, self.weight([hidden, y_size], name=name + '_out_weight')), self.bias(y_size, name=name + '_out_bias'), name=name + '_pred')
				]

				super().__init__(X_shape, y_size, layers, sess=sess, graph=graph, name=name)
				sess.run(tf.global_variables_initializer())

		self.graph = graph