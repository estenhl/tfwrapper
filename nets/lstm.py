import numpy as np
import tensorflow as tf
from .model import Model

DEFAULT_BATCH_SIZE = 1000
DEFAULT_EPOCHS = 5000

class LSTM(Model):
	def __init__(self, id, input_shape, classes):
		self.id = id
		self.build(input_shape, classes, 24)

	def build(self, input_shape, classes, num_hidden):
		seq_length, dimensions = input_shape

		with tf.Session() as sess:
			self.x = tf.placeholder(tf.float32, [None, seq_length, dimensions], name='x_placeholder')
			self.y = tf.placeholder(tf.float32, [None, classes], name='y_placeholder')

			num_hidden = 24
			cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)

			val, state = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)

			val = tf.transpose(val, [1, 0, 2])
			last = tf.gather(val, int(val.get_shape()[0]) - 1)

			weight = tf.Variable(tf.truncated_normal([num_hidden, int(self.y.get_shape()[1])]))
			bias = tf.Variable(tf.constant(0.1, shape=[self.y.get_shape()[1]]))

			self.pred = tf.nn.softmax(tf.matmul(last, weight) + bias)
			cross_entropy = -tf.reduce_sum(self.y * tf.log(tf.clip_by_value(self.pred,1e-10,1.0)))

			optimizer = tf.train.AdamOptimizer()
			self.minimize = optimizer.minimize(cross_entropy)

			correct_pred = tf.not_equal(tf.argmax(self.y, 1), tf.argmax(self.pred, 1))
			self.cost = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	def train(self, X, y, epochs=DEFAULT_EPOCHS, ids=None, refs=None):
		print(X.shape)
		train_X, train_y, val_X, val_y = self.split_data(X, y)
		train_X = np.reshape(train_X, [-1, train_X.shape[1], 1])
		val_X = np.reshape(val_X, [-1, val_X.shape[1], 1])
		batches = self.batch_data(DEFAULT_BATCH_SIZE, train_X, train_y)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(0, epochs):
				for batch in batches:
					inp, out = batch['x'], batch['y']
					sess.run(self.minimize,{self.x: inp, self.y: out})
				print("Epoch - " + str(i))
				incorrect = sess.run(self.cost,{self.x: val_X, self.y: val_y})
				print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))





