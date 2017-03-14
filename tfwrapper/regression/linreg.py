import tensorflow as tf

from .regression import RegressionModel

class LinearRegression(RegressionModel):
	def __init__(self, X_size, sess=None, name='LinearRegression'):
		if X_size != 1:
			raise NotImplementedError('Multivariable linear regression is not implemented')

		y_size = 1
		
		weight = tf.Variable(tf.random_normal([1]), name=name + '_weight')
		bias = tf.Variable(tf.random_normal([1]), name=name + '_bias')
		layer = lambda x: tf.add(tf.mul(x, weight), bias)
		
		super().__init__(X_size, y_size, layer, sess, name)

	def loss_function(self):
		return tf.reduce_sum(tf.pow(self.pred-self.y, 2))/(2 * self.batch_size)

	def optimizer_function(self):
		return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
