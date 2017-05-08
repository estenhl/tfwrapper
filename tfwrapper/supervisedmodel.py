import json
import math
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from .dataset import split_dataset
from tfwrapper.utils.data import batch_data
from tfwrapper.utils.exceptions import InvalidArgumentException

METAFILE_SUFFIX = 'tw'

class TFSession():
	def __init__(self, session=None, graph=None, init=False, variables={}):
		self.is_local_session = session is None
		self.session = session
		
		if session:
			self.graph = session.graph
			if init:
				self.session.run(tf.global_variables_initializer())
		elif graph:
			self.graph = graph
		else:
			self.graph = tf.Graph()

		if self.is_local_session:
			self.graph.as_default()
			self.session = tf.Session(graph=graph)
			if init:
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
	def bias(size, init='zeros', trainable=True, name='bias'):
		return SupervisedModel.weight([size], init=init, trainable=trainable, name=name)

	@staticmethod
	def weight(shape, init='truncated', stddev=0.02, trainable=True, name='weight'):
		if init == 'truncated':
			weight = tf.truncated_normal(shape, stddev=stddev)
		elif init == 'he_normal':
			# He et al., http://arxiv.org/abs/1502.01852
			fan_in, _ = SupervisedModel.compute_fan_in_out(shape)
			weight = tf.truncated_normal(shape, stddev=math.sqrt(2 / fan_in))
		elif init == 'xavier_normal':
			# Glorot & Bengio, AISTATS 2010 - http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
			fan_in, fan_out = SupervisedModel.compute_fan_in_out(shape)
			weight = tf.truncated_normal(shape, stddev=math.sqrt(2 / (fan_in + fan_out)))
		elif init == 'random':
			weight = tf.random_normal(shape)
		elif init == 'zeros':
			weight = tf.zeros(shape)
		else:
			raise NotImplementedError('Unknown initialization scheme %s' % str(init))

		return tf.Variable(weight, trainable=trainable, name=name)

	@staticmethod
	def compute_fan_in_out(weight_shape):
		if len(weight_shape) == 2:
			fan_in = weight_shape[0]
			fan_out = weight_shape[1]
		elif len(weight_shape) in {3, 4, 5}:
			# Assuming convolution kernels (1D, 2D or 3D).
			# TF kernel shape: (..., input_depth, depth)
			receptive_field_size = np.prod(weight_shape[:2])
			fan_in = weight_shape[-2] * receptive_field_size
			fan_out = weight_shape[-1] * receptive_field_size
		else:
			# No specific assumptions.
			fan_in = math.sqrt(np.prod(weight_shape))
			fan_out = math.sqrt(np.prod(weight_shape))
		return fan_in, fan_out

	@staticmethod
	def reshape(shape, name):
		return lambda x: tf.reshape(x, shape=shape, name=name)

	@staticmethod
	def out(*, inputs, outputs, init='truncated', trainable=True, name='pred'):
		weight_shape = [inputs, outputs]

		def create_layer(x):
			weight = SupervisedModel.weight(weight_shape, init=init, name=name + '/W', trainable=trainable)
			bias = SupervisedModel.bias(outputs, name=name + '/b')
			return tf.add(tf.matmul(x, weight), bias, name=name)

		return create_layer

	@staticmethod
	def relu(name):
		return lambda x: tf.nn.relu(x, name=name)

	@staticmethod
	def softmax(name):
		return lambda x: tf.nn.softmax(x, name=name)
		
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

		if verbose:
			print('Training ' + self.name + ' with ' + str(len(X)) + ' cases')

		with TFSession(sess, self.graph, init=True) as sess:
			for epoch in range(epochs):
				rand_idx = np.arange(len(X))
				np.random.shuffle(rand_idx)
				X_batches = batch_data(X[rand_idx], self.batch_size)
				y_batches = batch_data(y[rand_idx], self.batch_size)
				self.train_epoch(X_batches, y_batches, epoch, val_X=val_X, val_y=val_y, validate=validate, sess=sess, verbose=verbose)

			self.checkpoint_variables(sess)

	@abstractmethod
	def train_epoch(self, X_batches, y_batches, epoch_nr, val_X=None, val_y=None, validate=True, sess=None, verbose=False):
		raise NotImplementedError('SupervisedModel is a generic class')

	def predict(self, X, sess=None, verbose=False):
		with TFSession(sess, self.graph, variables=self.variables) as sess:
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

	@abstractmethod
	def validate(self, X, y, sess=None, verbose=False):
		raise NotImplementedError('SupervisedModel is a generic class')

	def save(self, filename, labels=[], sess=None):
		with TFSession(sess, self.graph, variables=self.variables) as sess:
			saver = tf.train.Saver()
			saver.save(sess, filename)

			metadata = {}
			metadata['name'] = self.name
			metadata['X_shape'] = self.X_shape
			metadata['y_size'] = self.y_size
			metadata['batch_size'] = self.batch_size
			metadata['labels'] = labels

			metadata_filename = '%s.%s' % (filename, METAFILE_SUFFIX)
			with open(metadata_filename, 'w') as f:
				f.write(json.dumps(metadata, indent=2))

	def load(self, filename, sess=None):
		with TFSession(sess, sess.graph) as sess:
			graph_path = filename + '.meta'
			saver = tf.train.Saver()
			saver.restore(sess, filename)

			self.graph = sess.graph
			self.X = sess.graph.get_tensor_by_name(self.name + '/X_placeholder:0')
			self.y = sess.graph.get_tensor_by_name(self.name + '/y_placeholder:0')
			self.lr = sess.graph.get_tensor_by_name(self.name + '/learning_rate_placeholder:0')
			self.pred = sess.graph.get_tensor_by_name(self.name + '/pred:0')

			self.checkpoint_variables(sess)

			# TODO: SHOULD USE METADATA, NOT SURE HOW THOUGH