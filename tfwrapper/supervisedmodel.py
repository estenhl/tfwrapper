import random
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from .dataset import split_dataset
from tfwrapper.utils.data import batch_data
from tfwrapper.utils.metrics import loss
from tfwrapper.utils.metrics import accuracy

class TFSession():
	def __init__(self, session=None, graph=None, init_vars=False, variables={}):
		self.has_local_session = (session is None)
		self.session = session
		self.graph = graph

		if self.has_local_session:
			print('Initing new session with graph ' + str(graph))
			self.session = tf.Session(graph=graph)

			if len(variables) > 0:
				for name in variables:
					print('Initialized ' + name)
					tf.Variable(variables[name], name=name)
			if init_vars:
				print('Initializing vars!')
				self.session.run(tf.global_variables_initializer())

	def __enter__(self):
		return self.session

	def __exit__(self, type, value, traceback):
		if self.has_local_session:
			self.session.close()


class SupervisedModel(ABC):
	graph = None
	variables = {}

	learning_rate = 0.1
	batch_size = 128

	def __init__(self, X_shape, y_size, layers, sess=None, graph=None, src=None, name='SupervisedModel'):
		if graph is None:
			if sess is not None:
				raise Exception('When a session is passed, a graph must be passed aswell')
			graph = tf.Graph()

		with TFSession(sess, graph) as sess:
			self.X_shape = X_shape
			self.y_size = y_size
			self.name = name
			self.input_size = np.prod(X_shape)
			self.output_size = y_size

			self.X = tf.placeholder(tf.float32, [None] + X_shape, name=self.name + '_X_placeholder')
			self.y = tf.placeholder(tf.float32, [None, y_size], name=self.name + '_y_placeholder')

			prev = self.X
			for layer in layers:
				prev = layer(prev)
			self.pred = prev

			self.loss = self.loss_function()
			self.optimizer = self.optimizer_function()

		self.graph = graph

	@abstractmethod
	def loss_function(self):
		raise NotImplementedError('SupervisedModel is a generic class')

	@abstractmethod
	def optimizer_function(self):
		raise NotImplementedError('SupervisedModel is a generic class')

	def weight(self, shape, name):
		return tf.Variable(tf.random_normal(shape), name=name)

	def bias(self, size, name):
		return tf.Variable(tf.random_normal([size]), name=name)

	def train(self, X, y, val_X=None, val_y=None, validate=True, epochs=5000, sess=None, verbose=False):
		assert len(X) == len(y)

		X = np.reshape(X, [-1] + self.X_shape)
		y = np.reshape(y, [-1, self.y_size])
		if val_X is None and validate:
			X, y, val_X, val_y = split_dataset(X, y)

		X_batches = batch_data(X, self.batch_size)
		y_batches = batch_data(y, self.batch_size)
		num_batches = len(X_batches)

		if verbose:
			print('Training ' + self.name + ' with ' + str(len(X)) + ' cases')

		with TFSession(sess, self.graph, init_vars=True) as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(epochs):
				for i in range(num_batches):
					sess.run(self.optimizer, feed_dict={self.X: X_batches[i], self.y: y_batches[i]})

				if verbose:
					loss, acc = self.validate(X_batches[-1], y_batches[-1], sess=sess, verbose=verbose)
					print('Epoch %d, train loss: %.3f, train acc: %2f' % (epoch + 1, loss, acc))

					if validate:
						loss, acc = self.validate(val_X, val_y, sess=sess, verbose=verbose)
						print('Epoch %d, val loss: %.3f, val acc: %2f' % (epoch + 1, loss, acc))

	def predict(self, X, sess=None, verbose=False):
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
		preds = self.predict(X, sess=sess, verbose=verbose)

		return loss(preds, y), accuracy(preds, y)

	def save(self, filename, sess=None):
		saver = tf.train.Saver()
		saver.save(sess, filename)

	def load(self, filename, sess=None):
		if sess is None:
			raise NotImplementedError('Loading outside a session is not implemented')

		with TFSession(sess, sess.graph) as sess:
			graph_path = filename + '.meta'
			saver = tf.train.import_meta_graph(graph_path)
			saver.restore(sess, filename)

			self.graph = sess.graph
			self.X = sess.graph.get_tensor_by_name(self.name + '_X_placeholder:0')
			self.y = sess.graph.get_tensor_by_name(self.name + '_y_placeholder:0')
			self.pred = sess.graph.get_tensor_by_name(self.name + '_pred:0')
