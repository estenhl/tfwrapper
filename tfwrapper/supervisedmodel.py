import random
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from .dataset import split_dataset
from tfwrapper.utils.data import batch_data
from tfwrapper.utils.metrics import loss
from tfwrapper.utils.metrics import accuracy
from tfwrapper.utils.exceptions import InvalidArgumentException

class TFSession():
	def __init__(self, session=None, graph=None, variables={}):
		self.is_local_session = session is None
		self.session = session
		
		if session:
			self.graph = session.graph
		elif graph:
			self.graph = graph
		else:
			self.graph = tf.Graph()

		if self.is_local_session:
			self.graph.as_default()
			self.session = tf.Session(graph=graph)
			self.session.run(tf.global_variables_initializer())
			if len(variables) > 0:
				for name in variables:
					variable = [v for v in tf.global_variables() if v.name == name][0]
					self.session.run(variable.assign(variables[name]))

	def __enter__(self):
		return self.session

	def __exit__(self, type, value, traceback):
		if self.is_local_session:
			self.session.close()


class SupervisedModel(ABC):
	graph = None
	variables = {}

	learning_rate = 0.1
	batch_size = 128

	def __init__(self, X_shape, y_size, layers, sess=None, name='SupervisedModel'):
		with TFSession(sess) as sess:
			self.X_shape = X_shape
			self.y_size = y_size
			self.name = name
			self.input_size = np.prod(X_shape)
			self.output_size = y_size

			self.X = tf.placeholder(tf.float32, [None] + X_shape, name=self.name + '/X_placeholder')
			self.y = tf.placeholder(tf.float32, [None, y_size], name=self.name + '/y_placeholder')
			self.lr = tf.placeholder(tf.float32, [], name=self.name + '/learning_rate_placeholder')

			prev = self.X
			for layer in layers:
				prev = layer(prev)
			self.pred = prev

			self.loss = self.loss_function()
			self.optimizer = self.optimizer_function()

			self.graph = sess.graph

	@abstractmethod
	def loss_function(self):
		raise NotImplementedError('SupervisedModel is a generic class')

	@abstractmethod
	def optimizer_function(self):
		raise NotImplementedError('SupervisedModel is a generic class')

	@staticmethod
	def reshape(shape, name):
		return lambda x: tf.reshape(x, shape=shape)

	@staticmethod
	def out(weight_shape, bias_size, name):
		return lambda x: tf.add(tf.matmul(x, tf.Variable(tf.random_normal(weight_shape))), tf.Variable(tf.random_normal([bias_size])), name=name)

	def checkpoint_variables(self, sess):
		for variable in tf.global_variables():
			self.variables[variable.name] = sess.run(variable)

	def train(self, X, y, val_X=None, val_y=None, validate=True, epochs=5000, sess=None, verbose=False):
		if not len(X) == len(y):
			raise InvalidArgumentException('X and y must be same length, not %d and %d' % (len(X), len(y)))
		
		if not (len(X.shape) >= 2 and list(X.shape[1:]) == self.X_shape):
			raise InvalidArgumentException('X with shape %s does not match given X_shape %s' % (str(X.shape), str(self.X_shape)))
		else:
			X = np.reshape(X, [-1] + self.X_shape)

		if not len(y.shape) == 2:
			raise InvalidArgumentException('y must be a onehot array')

		if not y.shape[1] == self.y_size:
			raise InvalidArgumentException('y with %d classes does not match given y_size %d' % (y.shape[1], self.y_size))
		else:
			y = np.reshape(y, [-1, self.y_size])

		if val_X is None and validate:
			X, y, val_X, val_y = split_dataset(X, y)

		X_batches = batch_data(X, self.batch_size)
		y_batches = batch_data(y, self.batch_size)
		num_batches = len(X_batches)

		if verbose:
			print('Training ' + self.name + ' with ' + str(len(X)) + ' cases')

		with TFSession(sess, self.graph) as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(epochs):
				for i in range(num_batches):
					sess.run(self.optimizer, feed_dict={self.X: X_batches[i], self.y: y_batches[i], self.lr: self.learning_rate})

				if verbose:
					loss, acc = self.validate(X_batches[-1], y_batches[-1], sess=sess, verbose=verbose)
					print('Epoch %d, train loss: %.3f, train acc: %.3f' % (epoch + 1, loss, acc))

					if validate:
						loss, acc = self.validate(val_X, val_y, sess=sess, verbose=verbose)
						print('Epoch %d, val loss: %.3f, val acc: %.3f' % (epoch + 1, loss, acc))

			self.checkpoint_variables(sess)

	def predict(self, X, sess=None, verbose=False):
		with TFSession(sess, self.graph, self.variables) as sess:
			X = np.reshape(X, [-1] + self.X_shape)
			batches = batch_data(X, self.batch_size)
			preds = None

			for batch in batches:
				batch_preds = sess.run(self.pred, feed_dict={self.X: batch})
				if preds is not None:
					preds = np.concatenate([preds, batch_preds])
				else:
					preds = batch_preds

		return preds

	def validate(self, X, y, sess=None, verbose=False):
		with TFSession(sess, self.graph, self.variables) as sess:
			preds = self.predict(X, sess=sess, verbose=verbose)

		return loss(preds, y), accuracy(preds, y)

	def save(self, filename, sess=None):
		with TFSession(sess, self.graph, self.variables) as sess:
			saver = tf.train.Saver()
			saver.save(sess, filename)

	def load(self, filename, sess=None):
		with TFSession(sess, sess.graph) as sess:
			graph_path = filename + '.meta'
			saver = tf.train.import_meta_graph(graph_path)
			saver.restore(sess, filename)

			self.graph = sess.graph
			self.X = sess.graph.get_tensor_by_name(self.name + '/X_placeholder:0')
			self.y = sess.graph.get_tensor_by_name(self.name + '/y_placeholder:0')
			self.lr = sess.graph.get_tensor_by_name(self.name + '/learning_rate_placeholder:0')
			self.pred = sess.graph.get_tensor_by_name(self.name + '/pred:0')

			self.checkpoint_variables(sess)
