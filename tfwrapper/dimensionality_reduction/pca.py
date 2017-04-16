import numpy as np 

from tfwrapper.utils.exceptions import IllegalStateException
from tfwrapper.utils.exceptions import InvalidArgumentException

from .dimensionality_reducer import DimensionalityReducer

class PCA(DimensionalityReducer):
	def __init__(self, dimensions, name='PCA'):
		super().__init__(dimensions, name=name)
		self.W = None

	def fit(self, X):
		if len(X.shape) != 2:
			raise InvalidArgumentException('Invalid dimensions for PCA, requires NxM matrix where N=num samples and M=dimensions')

		M = X.shape[1]
		if M < self.dimensions:
			raise InvalidArgumentException('Unable to perform PCA to a higher number of dimensions (From %d to %d)' % (M, self.dimensions))
		
		cov = np.cov(X, rowvar=False)
		eigenvalues, eigenvectors = np.linalg.eig(cov)
		pairs = list(zip(eigenvalues, eigenvectors))
		pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

		self.W = np.zeros((M, self.dimensions))
		for i in range(M):
			for j in range(self.dimensions):
				self.W[i][j] = pairs[j][1][i]

	def transform(self, X):
		if self.W is None:
			raise IllegalStateException('PCA must be fit before it can transform')

		return np.dot(X, self.W)