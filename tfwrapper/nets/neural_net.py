import sys
import numpy as np
import tensorflow as tf

from time import process_time
from tfwrapper.metrics import accuracy
from tfwrapper import TFSession
from tfwrapper import SupervisedModel

class NeuralNet(SupervisedModel):
	def __init__(self, X_shape, classes, layers, sess=None, name='NeuralNet'):

		with TFSession(sess) as sess:
			super().__init__(X_shape, classes, layers, sess=sess, name=name)

			self.accuracy = self.accuracy_function()

			self.graph = sess.graph


	def loss_function(self):
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y, name=self.name + '/softmax'), name=self.name + '/loss')

	def optimizer_function(self):
		return tf.train.AdamOptimizer(learning_rate=self.lr, name=self.name + '/adam').minimize(self.loss, name=self.name + '/optimizer')

	def accuracy_function(self):
		correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
		return tf.reduce_mean(tf.cast(correct_pred, tf.float32), name=self.name + '/accuracy')

	@staticmethod
	def fullyconnected(*, inputs, outputs, trainable=True, activation='relu', name='fullyconnected'):
		weight_shape = [inputs, outputs]
		weight_name = name + '/W'
		bias_name = name + '/b'

		def create_layer(x):
			weight = NeuralNet.weight(weight_shape, name=weight_name, trainable=trainable)
			bias = NeuralNet.bias(outputs, name=bias_name, trainable=trainable)

			fc = tf.reshape(x, [-1, inputs], name=name + '/reshape')
			fc = tf.add(tf.matmul(fc, weight), bias, name=name + '/add')
			
			if activation == 'relu':
				fc = tf.nn.relu(fc, name=name)
			elif activation == 'softmax':
				fc = tf.nn.softmax(fc, name=name)
			else:
				raise NotImplementedError('%s activation is not implemented (Valid: [\'relu\', \'softmax\'])' % activation)

			return fc

		return create_layer


	@staticmethod
	def dropout(dropout, name='dropout'):
		return lambda x: tf.nn.dropout(x, dropout, name=name)


	def load(self, filename, sess=None):
		with TFSession(sess, self.graph, self.variables) as sess:
			super().load(filename, sess=sess)

			self.loss = sess.graph.get_tensor_by_name(self.name + '/loss:0')
			self.accuracy = sess.graph.get_tensor_by_name(self.name + '/accuracy:0')

	def train_epoch(self, X_batches, y_batches, epoch_nr, val_X=None, val_y=None, validate=True, sess=None, verbose=False):
		with TFSession(sess, self.graph) as sess:
			num_batches = len(X_batches)
			num_items = num_batches * self.batch_size - (self.batch_size - len(X_batches[-1]))
			epoch_loss_avg = 0
			epoch_acc_avg = 0
			epoch_time = 0
			for i in range(num_batches):
				start_batch_time = process_time()
				_, loss_val, acc_val = sess.run([self.optimizer, self.loss, self.accuracy],
												feed_dict={self.X: X_batches[i], self.y: y_batches[i],
														   self.lr: self.learning_rate})
				epoch_time += process_time() - start_batch_time
				epoch_loss_avg += loss_val / num_batches
				epoch_acc_avg += acc_val / num_batches
				if verbose:
					# Display a summary of this batch reuing the same terminal line
					sys.stdout.write('\033[K')  # erase previous line in case the new line is shorter
					sys.stdout.write('\riter: {:0{}}/{} | batch loss: {:.5} - acc {:.5}' \
									 ' | time: {:.3}s'.format(i * self.batch_size + len(X_batches[i]),
															  len(str(num_items)), num_items, loss_val, acc_val, epoch_time))
					sys.stdout.flush()

			if verbose:
				epoch_summary = '\nEpoch: \033[1m\033[32m{}\033[0m\033[0m | avg batch loss: \033[1m\033[32m{:.5}\033[0m\033[0m' \
								' - avg acc: {:.5}'.format(epoch_nr + 1, epoch_loss_avg, epoch_acc_avg)
				if validate and (val_X is not None) and (val_y is not None):
					loss_val, acc_val = self.validate(val_X, val_y, sess=sess, verbose=verbose)
					epoch_summary += ' | val_loss: \033[1m\033[32m{:.5}\033[0m\033[0m - val_acc: {:.5}'.format(loss_val, acc_val)
				print(epoch_summary, '\n')


	def validate(self, X, y, sess=None, verbose=False):
		with TFSession(sess, self.graph, variables=self.variables) as sess:
			preds = self.predict(X, sess=sess, verbose=verbose)
			loss_val = sess.run(self.loss, feed_dict={self.pred: preds, self.y: y})
		return loss_val, accuracy(preds, y)
