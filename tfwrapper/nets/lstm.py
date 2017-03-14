import tensorflow as tf

from .neural_net import NeuralNet

class LSTM(NeuralNet):
	def __init__(self, X_shape, classes, hidden, sess=None, name='LSTM'):
		self.hidden = hidden
		super().__init__(X_shape, classes, [self.layer], sess=sess, name=name)

	def layer(self, X):
		cell = tf.nn.rnn_cell.LSTMCell(self.hidden, state_is_tuple=True)

		val, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

		val = tf.transpose(val, [1, 0, 2])
		last = tf.gather(val, int(val.get_shape()[0]) - 1)

		weight = tf.Variable(tf.truncated_normal([self.hidden, int(self.y.get_shape()[1])]))
		bias = tf.Variable(tf.constant(0.1, shape=[self.y.get_shape()[1]]))

		return tf.nn.softmax(tf.matmul(last, weight) + bias)
		
	def optimizer_function(self):
		cross_entropy = -tf.reduce_sum(self.y * tf.log(tf.clip_by_value(self.pred,1e-10,1.0)))

		return tf.train.AdamOptimizer().minimize(cross_entropy)

	def accuracy_function(self):
		return lambda x: 0.0
