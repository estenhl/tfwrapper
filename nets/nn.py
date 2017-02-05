import os
import random
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128

class NN(ABC):
	def __init__(self, id, input_shape, classes, class_weights=None):
		self.id = id
		self.input_shape = input_shape
		self.classes = classes
		self.variables = {}

		if len(input_shape) == 3:
			height, width, channels = input_shape
			input_size = height * width * channels
		else:
			input_size = input_shape[0]

		self.graph = tf.Graph()
		with tf.Session(graph=self.graph) as sess:
			self.x = tf.placeholder(tf.float32, [None, input_size], name='x_placeholder')
			self.y = tf.placeholder(tf.float32, [None, classes], name='y_placeholder')

			weights = self.weights()
			biases = self.biases()

			if class_weights is None:
				class_weights = np.ones(classes) / np.sum(np.ones(classes))

			self.pred, self.layers = self.net(self.x, input_shape, weights, biases)
			self.weighted_pred = tf.mul(self.pred, class_weights, name='weighted_pred')

			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.weighted_pred, self.y, name='softmax'), name='reduce_mean')
			self.optimizer = tf.train.AdamOptimizer(learning_rate=DEFAULT_LEARNING_RATE, name='adam').minimize(self.cost)

			correct_pred = tf.equal(tf.argmax(self.weighted_pred, 1), tf.argmax(self.y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	@abstractmethod
	def net(self, x, input_shape, weights, biases):
		raise NotImplementedError

	def split_data(self, X, y=None):
		batches = []

		for i in range(0, int(len(X) / DEFAULT_BATCH_SIZE) + 1):
			start = (i * DEFAULT_BATCH_SIZE)
			end = min((i + 1) * DEFAULT_BATCH_SIZE, len(X))
			batch_X = X[start:end]
			if y is None:
				batches.append({'x': batch_X})
			else:
				batch_y = y[start:end]
				batches.append({'x': batch_X, 'y': batch_y})

		return batches

	def checkpoint_variables(self, sess):
		for var in tf.global_variables():
			self.variables[var] = sess.run(var)

	def initialize_session(self):
		sess = tf.Session(graph=self.graph)

		if len(self.variables) > 0:
			for var in self.variables:
				sess.run(var.assign(self.variables[var]))

		return sess

	def train_epoch(self, sess, batches, steps=1):
		for i, batch in enumerate(batches):
			sess.run(self.optimizer, feed_dict={self.x: batch['x'], self.y: batch['y']})
					
			if i == len(batches) - 1:
				loss, acc = sess.run([self.cost, self.accuracy], feed_dict={self.x: batch['x'], self.y: batch['y']})
					
				print("Training step " + str(steps * DEFAULT_BATCH_SIZE) + ", training loss: " + \
					"{:.2f}".format(loss) + ", training acc.: {:.4f}".format(acc))
			steps += 1

		return steps

	def validate_epoch(self, sess, val_batches, total_len):
		loss = 0.0
		acc = 0.0

		for val_batch in val_batches:
			batch_loss, batch_acc = sess.run([self.cost, self.accuracy], feed_dict={self.x: val_batch['x'], self.y: val_batch['y']})
			loss += batch_loss * len(val_batch['x'])
			acc += batch_acc * len(val_batch['x'])

			v = val_batch

		loss /= total_len
		acc /= total_len
		
		return loss, acc

	def fit(self, train_X, train_y, val_X, val_y, epochs=DEFAULT_EPOCHS):
		if len(self.input_shape) == 3:
			height, width, channels = self.input_shape
			input_size = height * width * channels
		else:
			input_size = self.input_shape[0]

		train_X = np.reshape(train_X, [-1, input_size])
		val_X = np.reshape(val_X, [-1, input_size])

		batches = self.split_data(train_X, train_y)
		val_batches = self.split_data(val_X, val_y)

		print('Started training with ' + str(len(train_X)) + ' images')
		with self.initialize_session() as sess:
			sess.run(tf.global_variables_initializer())
			steps = 1
			for epoch in range(0, epochs):
				random.shuffle(batches)

				steps = self.train_epoch(sess, batches, steps=steps)
				loss, acc = self.validate_epoch(sess, val_batches, len(val_X))
				print("Epoch " + str(epoch + 1) + ", val loss: " + \
					"{:.2f}".format(loss) + ", val acc.: " + \
					"{:.4f}".format(acc))

			self.checkpoint_variables(sess)
			
	def write_layers(self, path):
		f = open(path, 'w')

		for layer in self.layers:
			f.write(layer['name'] + ', ' + layer['size'] + '\n')

		f.close()

	def restore_layers(self, path):
		f = open(path, 'r')

		for line in f.readlines():
			name, size = line.split(', ')
			self.layers.append({'name': name, 'size': size})

		f.close()

	def save(self, path):
		model_path = os.path.join(path, 'model.ckpt')
		layers_path = os.path.join(path, 'layers.txt')
		with self.initialize_session() as sess:
			saver = tf.train.Saver(tf.global_variables())
			saver.save(sess, model_path)
			self.write_layers(layers_path)
			
			return True

		return False

	def load(self, path):
		model_path = os.path.join(path, 'model.ckpt')
		graph_path = model_path + '.meta'
		layers_path = os.path.join(path, 'layers.txt')
		with tf.Session(graph=self.graph) as sess:
			sess.run(tf.global_variables_initializer())
			saver = tf.train.import_meta_graph(graph_path)
			saver.restore(sess, path)
			self.restore_layers(layers_path)

			self.checkpoint_variables(sess)

	def predict(self, X):
		height, width, channels = self.input_shape
		input_size = height * width * channels
		X = np.reshape(X, (-1, input_size))
		batches = self.split_data(X)
		predictions = np.zeros(0)

		with self.initialize_session() as sess:
			for batch in batches:
				batch_preds = sess.run(self.pred, feed_dict={self.x: batch['x']})
				if len(predictions) == 0:
					predictions = batch_preds
				else:
					predictions = np.concatenate((predictions, batch_preds))
		
		return predictions

	def extract_features(self, X, layer_name):
		print('Extracting features from ' + layer_name)

		height, width, channels = self.input_shape
		input_size = height * width * channels
		X = np.reshape(X, (-1, input_size))
		batches = self.split_data(X)
		features = np.zeros(0)

		with self.initialize_session() as sess:
			layer = sess.graph.get_tensor_by_name(layer_name)
			if layer is not None:
				for batch in batches:
					batch_preds = sess.run(layer, feed_dict={self.x: batch['x']})
					if len(features) == 0:
						features = batch_preds
					else:
						features = np.concatenate((features, batch_preds))
			else:
				print('Graph ' + self.id + ' has no layer ' + layer_name)

		return features