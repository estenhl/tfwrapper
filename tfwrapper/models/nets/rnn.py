from tfwrapper.layers import recurring

from .neural_net import NeuralNet

class RNN(NeuralNet):
	def __init__(self, seq_shape, seq_length, num_hidden, classes, sess=None, name='RNN'):
		X_shape = seq_shape + [seq_length]
		layers = [recurring(seq_shape, seq_length, num_hidden, classes, name=name)]

		super().__init__(X_shape, classes, layers, sess=sess, name=name)

