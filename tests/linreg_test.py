import pytest
import random
import numpy as np
import tensorflow as tf

from utils.data import normalize
from regression import LinearRegression
"""
def train_linreg_1d(dims=1, train_size=100, epochs=30):
	X = np.asarray([[random.randint(0, 5) for j in range(dims)] for i in range(train_size)])
	factors = [random.randint(1, dims) for i in range(dims)]
	y = np.asarray([[sum([x[i] * factors[i] for i in range(dims)])] for x in X])
	train_len = int(len(X) * 0.8)

	X = normalize(X)
	y = normalize(y)

	sess = tf.Session()
	reg = LinearRegression(dims, sess=sess, name='LinRegTest')
	reg.train(X[:train_len], y[:train_len], epochs=epochs, sess=sess)

	return sess, reg, X[train_len:], y[train_len:]

def test_linreg_prediction():
	val_size = 500
	sess, reg, val_X, val_y = train_linreg_1d(train_size=100, epochs=1)
	pred_y = reg.predict(val_X, sess=sess)
	sess.close()

	assert pred_y is not None
	assert len(pred_y) == len(val_y)
	assert pred_y.dtype == float

def test_linreg_validation():
	sess, reg, val_X, val_y = train_linreg_1d(train_size=100, epochs=1)
	loss, _ = reg.validate(val_X, val_y, sess=sess)
	sess.close()

	assert loss is not None
	assert loss.dtype == float

def calculate_linreg_acc(dims=1, epochs=10):
	train_size = 50000
	val_size = 500
	sess, reg, val_X, val_y = train_linreg_1d(dims=dims, train_size=train_size, epochs=epochs)
	pred_y = reg.predict(val_X, sess=sess)

	loss = 0
	for i in range(0, val_size):
		loss += abs(pred_y[i] - val_y[i])
	
	avg_loss = loss / val_size
	return avg_loss

def test_linreg_accuracy_1d():
	loss = calculate_linreg_acc()
	assert loss < 0.1
"""