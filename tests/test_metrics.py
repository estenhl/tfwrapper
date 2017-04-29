import numpy as np
import tensorflow as tf

from tfwrapper.metrics import loss
from tfwrapper.metrics import accuracy
from tfwrapper.metrics import confusion_matrix
from tfwrapper.metrics.distance import euclidean
from tfwrapper.metrics.entropy import kullback_leibler

def create_data():
	y = np.asarray([
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1]
	]).astype(float)

	yhat = np.asarray([
		[0.3, 0.1, 0.2],
		[0.1, 0.3, 0.2],
		[0.2, 0.1, 0.3]
	])

	loss = 0
	correct = 0
	conf_matrix = np.zeros((3, 3))
	for i in range(len(y)):
		for j in range(len(y[i])):
			loss += abs(y[i][j] - yhat[i][j])
		if np.argmax(y[i]) == np.argmax(yhat[i]):
			correct += 1
		conf_matrix[np.argmax(y[i])][np.argmax(yhat[i])] = conf_matrix[np.argmax(y[i])][np.argmax(yhat[i])] + 1

	return y, yhat, loss / np.prod(y.shape) , correct / len(y), conf_matrix

def test_loss():
	y, yhat, correct_loss, _, _ = create_data()

	assert correct_loss == loss(y, yhat)

def test_accuracy():
	y, yhat, _, correct_acc, _ = create_data()

	assert correct_acc == accuracy(y, yhat)

def test_confusion_matrix():
	y, yhat, _, _, correct_conf_matrix = create_data()

	assert np.array_equal(correct_conf_matrix, confusion_matrix(y, yhat))

def test_euclidean_2d():
	x1 = np.asarray([0, 0])
	x2 = np.asarray([10, 10])
	dist = np.sqrt(10**2 + 10**2)

	assert dist == euclidean(x1, x2)

def test_euclidean_3d():
	x1 = np.asarray([0, 0, 0])
	x2 = np.asarray([10, 10, 10])
	dist = np.sqrt(10**2 + 10**2 + 10**2)

	assert dist == euclidean(x1, x2)

def test_kullback_leibler():
	P_init = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
	Q_init = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

	divergence = 0
	for i in range(3):
		for j in range(3):
			if i == j:
				continue

			divergence += P_init[i][j] * np.log(P_init[i][j] / Q_init[i][j])

	with tf.Session() as sess:
		name = 'KullbackLeiblerTest'
		P = tf.Variable(initial_value=P_init, name=name + '/P')
		Q = tf.Variable(initial_value=Q_init, name=name + '/Q')
		sess.run(tf.global_variables_initializer())

		assert 0.01 > np.abs(divergence - sess.run(kullback_leibler(P, Q, sess=sess)))

