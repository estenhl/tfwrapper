import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from supervisedmodel import TFSession
from supervisedmodel import SupervisedModel

class RegressionModel(SupervisedModel):
	def __init__(self, X_size, y_size, layer, sess=None, name='Regression'):
		super().__init__([X_size], y_size, [layer], sess=sess, name=name)

	def validate(self, X, y, sess=None):
		assert len(X) == len(y)

		X_batches = self.batch_data(X)
		y_batches = self.batch_data(y)
		num_batches = len(X_batches)
		loss = 0

		with TFSession(sess, self.graph) as sess:
			for i in range(num_batches):
				batch_loss = sess.run(self.loss, feed_dict={self.X: X_batches[i], self.y: y_batches[i]})
				loss += batch_loss

		return loss, 0.0
