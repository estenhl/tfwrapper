from tfwrapper.metrics.distance import euclidean

from .dimensionality_reducer import DimensionalityReducer

class TSNE(DimensionalityReducer):
	def __init__(self, dimensions, name='t-SNE'):
		super().__init__(dimensions, name=name)

	def fit(self, data):
		raise NotImplementedException('t-SNE not implemented')

	def transform(self, data):
		raise NotImplementedException('t-SNE not implemented')