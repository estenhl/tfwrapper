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

		with TFSession(sess, graph) as sess:
			super().__init__(X_shape, classes, layers, sess=sess, graph=graph, name=name)

			correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
			self.accuracy = self.accuracy_function(correct_pred)

		self.graph = graph

	def loss_function(self):
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y, name=self.name + '_softmax'), name=self.name + '_loss')

	def optimizer_function(self):
		return tf.train.AdamOptimizer(learning_rate=self.learning_rate, name=self.name + '_adam').minimize(self.loss, name=self.name + '_optimizer')

	def accuracy_function(self, correct_pred):
		return tf.reduce_mean(tf.cast(correct_pred, tf.float32), name=self.name + '_accuracy')

	def fullyconnected(self, *, input_size, output_size, name='fullyconnected'):
		weight_shape = [input_size, output_size]
		weight_name = name + '/weights'
		bias_name = name + '/biases'

		def create_layer(x):
			weight = tf.Variable(tf.random_normal(weight_shape), name=weight_name)
			bias = tf.Variable(tf.random_normal([output_size]), name=bias_name)

			fc = tf.reshape(x, [-1, weight.get_shape().as_list()[0]], name=name + '_reshape')
			fc = tf.add(tf.matmul(fc, weight), output_size, name=name + '_add')
			fc = tf.nn.relu(fc, name=name)

			return fc

		return create_layer

	def dropout(self, dropout, name='dropout'):
		return lambda x: tf.nn.dropout(x, dropout, name=name)

	def load(self, filename, sess=None):
		if sess is None:
			raise NotImplementedError('Loading outside a session is not implemented')

		with TFSession(sess) as sess:
			super().load(filename, sess=sess)
			self.loss = sess.graph.get_tensor_by_name(self.name + '_loss:0')
			self.accuracy = sess.graph.get_tensor_by_name(self.name + '_accuracy:0')
