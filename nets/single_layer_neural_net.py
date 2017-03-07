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
					lambda x: self.fullyconnected(x, self.weights(X_size, hidden, y_size)[0], self.biases(hidden, y_size)[0], name=name + '_hidden'),
					lambda x: tf.add(tf.matmul(x, self.weights(X_size, hidden, y_size)[1]), self.biases(hidden, y_size)[1], name=name + '_pred')
				]

				super().__init__(X_shape, y_size, layers, sess=sess, graph=graph, name=name)
				sess.run(tf.global_variables_initializer())

		self.graph = graph

	def weights(self, X_size, hidden, y_size):
		return [
			tf.Variable(tf.random_normal([X_size, hidden]), name=self.name + '_hidden_weight'),
			tf.Variable(tf.random_normal([hidden, y_size]), name=self.name + '_out_weight')
		]

	def biases(self, hidden, y_size):
		return [
			tf.Variable(tf.random_normal([hidden]), name=self.name + '_hidden_bias'),
			tf.Variable(tf.random_normal([y_size]), name=self.name + '_out_bias')
		]