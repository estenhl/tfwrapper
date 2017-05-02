import numpy as np

from tfwrapper.utils.exceptions import InvalidArgumentException

from .dimensionality_reducer import DimensionalityReducer

class LDA():
	def __init__(self, dimensions):
		self.dimensions = dimensions

	def fit(self, X, y):
		if len(X.shape) != 2:
			raise InvalidArgumentException('Invalid dimensions for LDA, requires NxM matrix where N=num samples and M=dimensions')

		N, M = X.shape
		if M < self.dimensions:
			raise InvalidArgumentException('Unable to perform LDA to a higher number of dimensions (From %d to %d)' % (M, self.dimensions))
		
		if len(y.shape) != 2 or y.shape[1] == 1:
			raise InvalidArgumentException('LDA requires onehot y matrix')

		num_classes = (max(np.argmax(y_) for y_ in y[:,1])) + 1
		classes = [[] for i in range(num_classes)]

		for i in range(N):
			classes[np.argmax(y[i])].append(X[i])

		for i in range(num_classes):
			classes[i] = np.asarray(classes[i])

		means = np.zeros((num_classes, M))
		for i in range(num_classes):
			for j in range(M):
				means[i][j] = np.mean(classes[i][:,j])
