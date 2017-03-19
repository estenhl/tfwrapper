import numpy as np

from tfwrapper.utils.metrics import loss
from tfwrapper.utils.metrics import accuracy
from tfwrapper.utils.metrics import confusion_matrix

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

	return y, yhat, loss, correct / len(y), conf_matrix

def test_loss():
	y, yhat, correct_loss, _, _ = create_data()

	assert correct_loss == loss(y, yhat)

def test_accuracy():
	y, yhat, _, correct_acc, _ = create_data()

	assert correct_acc == accuracy(y, yhat)

def test_confusion_matrix():
	y, yhat, _, _, correct_conf_matrix = create_data()

	assert np.array_equal(correct_conf_matrix, confusion_matrix(y, yhat))
