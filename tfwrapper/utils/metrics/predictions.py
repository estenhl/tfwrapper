import numpy as np

def loss(y, yhat):
	loss = 0

	for i in range(len(y)):
		for j in range(len(y[i])):
			loss += abs(y[i][j] - yhat[i][j])

	return loss / np.prod(y.shape)

def accuracy(y, yhat):
	correct = 0

	for i in range(len(y)):
		if np.argmax(y[i]) == np.argmax(yhat[i]):
			correct += 1

	return correct / len(y)

def confusion_matrix(y, yhat):
	matrix = np.zeros((y.shape[1], y.shape[1]))

	for i in range(len(y)):
		matrix[np.argmax(y[i])][np.argmax(yhat[i])] = matrix[np.argmax(y[i])][np.argmax(yhat[i])] + 1

	return matrix