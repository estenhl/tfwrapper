from .dimensionality_reducer import dimensionality_reducer

class PCA(DimensionalityReducer):
	def __init__(self, dimensions, name='PCA'):
		self.dimensions = dimensions
		self.name = name

	def fit(self, data):
		raise NotImplementedException('PCA not implemented')

	def transform(self, data):
		raise NotImplementedException('PCA not implemented')