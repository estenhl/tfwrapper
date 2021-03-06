import numpy as np
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.nets import NeuralNet

class SingleLayerNeuralNet(NeuralNet):
	def __init__(self, X_shape, y_size, hidden, sess=None, name='SingleLayerNeuralNet'):
		X_size = np.prod(X_shape)

		layers = [
			lambda x: self.fullyconnected(x, self.weights(X_size, hidden, y_size)[0], self.biases(hidden, y_size)[0]),
			lambda x: tf.add(tf.matmul(x, self.weights(X_size, hidden, y_size)[1]), self.biases(hidden, y_size)[1])
		]

		super().__init__(X_shape, y_size, layers, sess=sess, name=name)
		sess.run(tf.global_variables_initializer())

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