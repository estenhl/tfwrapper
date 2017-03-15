import numpy as np
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper import SupervisedModel

class NeuralNet(SupervisedModel):
	def __init__(self, X_shape, classes, layers, sess=None, graph=None, name='NeuralNet'):
		if graph is None:
			if sess is not None:
				raise Exception('When a session is passed, a graph must be passed aswell')
			graph = tf.Graph()

		with graph.as_default():
			with TFSession(sess, graph) as sess:
				super().__init__(X_shape, classes, layers, sess=sess, graph=graph, name=name)

				self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1), name=name + '_correct_pred')
				self.accuracy = self.accuracy_function()

		self.graph = graph

	def loss_function(self):
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y, name=self.name + '_softmax'), name=self.name + '_loss')

	def optimizer_function(self):
		return tf.train.AdamOptimizer(learning_rate=self.learning_rate, name=self.name + '_adam').minimize(self.loss, name=self.name + '_optimizer')

	def accuracy_function(self):
		return tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name=self.name + '_accuracyKKK')

	def fullyconnected(self, prev, weight, bias, name=None):
		fc = tf.reshape(prev, [-1, weight.get_shape().as_list()[0]], name=name + '_reshape')
		fc = tf.add(tf.matmul(fc, weight), bias, name=name + '_add')
		fc = tf.nn.relu(fc, name=name)

		return fc

	def validate(self, X, y, sess=None, verbose=False):
		assert len(X) == len(y)

		X = np.reshape(X, [-1] + self.X_shape)
		y = np.reshape(y, [-1, self.y_size])

		X_batches = self.batch_data(X)
		y_batches = self.batch_data(y)
		num_batches = len(X_batches)
		loss = 0
		acc = 0

		with self.graph.as_default():
			with TFSession(sess, self.graph) as sess:
				for i in range(num_batches):
					batch_loss, batch_acc = sess.run([self.loss, self.accuracy], feed_dict={self.X: X_batches[i], self.y: y_batches[i]})
					print('VALBATCH: ' + str(batch_loss) + ' ' + str(batch_acc))
					loss += batch_loss
					acc += batch_acc * (len(X_batches[i]) / len(X))

		return loss / len(X), acc

	def load(self, filename, sess=None):
		if sess is None:
			raise NotImplementedError('Loading outside a session is not implemented')
			
		with TFSession(sess) as sess:
			super().load(filename, sess=sess)
			self.loss = sess.graph.get_tensor_by_name(self.name + '_loss:0')
			self.accuracy = sess.graph.get_tensor_by_name(self.name + '_accuracy:0')