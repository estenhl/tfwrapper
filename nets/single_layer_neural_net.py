import numpy as np
import tensorflow as tf

from .neural_net import NeuralNet

class SingleLayerNeuralNet(NeuralNet):
	def __init__(self, X_shape, y_size, hidden, sess=None, name='SingleLayerNeuralNet'):
		X_size = np.prod(X_shape)
		weights = [
			tf.Variable(tf.random_normal([X_size, hidden]), name=name + '_hidden_weight'),
			tf.Variable(tf.random_normal([hidden, y_size]), name='bd1')
		]
		biases = [
			tf.Variable(tf.random_normal([hidden]), name=name + '_hidden_bias'),
			tf.Variable(tf.random_normal([y_size]), name=name + '_hidden_bias')
		]
		layers = [
			lambda x: self.fullyconnected(x, weights[0], biases[0]),
			lambda x: self.fullyconnected(x, weights[1], biases[1])
		]

		super().__init__(X_shape, y_size, layers, sess=sess, name=name)

