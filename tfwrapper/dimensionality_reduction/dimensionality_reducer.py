from abc import ABC, abstractmethod

class DimensionalityReducer(ABC):
	def __init__(self, dimensions, name='DimensionalityReducer'):
		self.dimensions = dimensions
		self.name = name

	@abstractmethod
	def fit(self, data):
		raise NotImplementedException('DimensionalityReducer is an abstract class')

	@abstractmethod
	def transform(self, data):
		raise NotImplementedException('DimensionalityReducer is an abstract class')