import random
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

class TFSession():
	def __init__(self, session=None, graph=None, init_vars=False):
		self.has_local_session = (session is None)
		self.session = session
		self.graph = graph

		if self.has_local_session:
			print('Initing new session with graph ' + str(graph))
			self.session = tf.Session(graph=graph)
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

	learning_rate = 0.1
	batch_size = 128

	def __init__(self, X_shape, y_size, layers, sess=None, name='SupervisedModel'):
		self.X_shape = X_shape
		self.y_size = y_size
		self.name = name
		self.input_size = np.prod(X_shape)
		self.output_size = y_size

		self.X = tf.placeholder(tf.float32, [None, self.input_size], name='X_placeholder')
		self.y = tf.placeholder(tf.float32, [None, y_size], name='y_placeholder')

		prev = self.X
		for layer in layers:
			prev = layer(prev)
		self.pred = prev

		self.loss = self.loss_function()
		self.optimizer = self.optimizer_function()

	@abstractmethod
	def loss_function(self):
		raise NotImplementedError('SupervisedModel is a generic class')
	
	@abstractmethod
	def optimizer_function(self):
		raise NotImplementedError('SupervisedModel is a generic class')
		
	def split_data(self, X, y, val_split=0.8):
		train_len = int(len(X) * val_split)
		train_X = X[:train_len]
		train_y = y[:train_len]
		val_X = X[train_len:]
		val_y = y[train_len:]

		return train_X, train_y, val_X, val_y
		
	def batch_data(self, data):
		batches = []

		for i in range(0, int(len(data) / self.batch_size) + 1):
			start = (i * self.batch_size)
			end = min((i + 1) * self.batch_size, len(data))
			batches.append(data[start:end])

		return batches

	def train(self, X, y, val_X=None, val_y=None, validate=True, epochs=5000, sess=None, verbose=True):
		print('Training with sess: ' + str(sess) + ', graph: ' + str(sess.graph))
		assert len(X) == len(y)

		X = np.reshape(X, [-1, self.input_size])
		y = np.reshape(y, [-1, self.y_size])
		if val_X == None and validate:
			X, y, val_X, val_y = self.split_data(X, y)
			
		X_batches = self.batch_data(X)
		y_batches = self.batch_data(y)
		num_batches = len(X_batches)
		
		if verbose:
			print('Training ' + self.name + ' with ' + str(len(X)) + ' cases')

		#sess.run(tf.global_variables_initializer())
		for epoch in range(epochs):
			for i in range(num_batches):
				sess.run(self.optimizer, feed_dict={self.X: X_batches[i], self.y: y_batches[i]})

			if verbose:			
				train_loss, train_acc = self.validate(X_batches[num_batches - 1], y_batches[num_batches - 1], sess=sess)
				val_loss, val_acc = self.validate(val_X, val_y, sess=sess)
				print('Epoch %d, train loss: %.3f, train acc: %2f, val loss: %.3f, val acc: %2f' % (epoch + 1, train_loss, train_acc, val_loss, val_acc))

	def predict(self, X, sess=None):
		batches = self.batch_data(X)
		preds = np.array([])

		with TFSession(sess, self.graph) as sess:
			for batch in batches:
				preds = np.concatenate([preds, sess.run(self.pred, feed_dict={self.X: batch}).flatten()])

		return preds

	def save(self, filename):
		print('Saving to ' + filename)

	def load(self, filename):
		print('Loading from ' + filename)