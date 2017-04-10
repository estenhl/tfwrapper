import numpy as np
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper import SupervisedModel

class NeuralNet(SupervisedModel):
	def __init__(self, X_shape, classes, layers, sess=None, name='NeuralNet'):

		with TFSession(sess) as sess:
			super().__init__(X_shape, classes, layers, sess=sess, name=name)

			correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
			self.accuracy = self.accuracy_function(correct_pred)

			self.graph = sess.graph

	def loss_function(self):
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y, name=self.name + '/softmax'), name=self.name + '/loss')

	def optimizer_function(self):
		return tf.train.AdamOptimizer(learning_rate=self.lr, name=self.name + '_adam').minimize(self.loss, name=self.name + '/optimizer')

	def accuracy_function(self, correct_pred):
		return tf.reduce_mean(tf.cast(correct_pred, tf.float32), name=self.name + '/accuracy')

	@staticmethod
	def fullyconnected(*, input_size, output_size, name='fullyconnected'):
		weight_shape = [input_size, output_size]
		weight_name = name + '_W'
		bias_name = name + '_b'

		def create_layer(x):
			weight = tf.Variable(tf.truncated_normal(weight_shape, stddev=0.02), name=weight_name)
			bias = tf.Variable(tf.zeros([output_size]), name=bias_name)

			fc = tf.reshape(x, [-1, weight.get_shape().as_list()[0]], name=name + '/reshape')
			fc = tf.add(tf.matmul(fc, weight), bias, name=name + '/add')
			fc = tf.nn.relu(fc, name=name)

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