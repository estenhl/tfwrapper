from .dimensionality_reducer import dimensionality_reducer

class PCA(DimensionalityReducer):
	def __init__(self, dimensions, name='t-SNE'):
		self.dimensions = dimensions
		self.name = name

	def calculate_conditional_probability(self, data, i, j, )

	def calculate_probability(self, data, i, j, N=None):
		if N is None:
			N = len(data)

		Pij = self.calculate_conditional_probability(data, i, j)
		Pji = self.calculate_conditional_probability(data, j, i)

		return (Pij + Pji) / (2 * N)

	def fit(self, data):
		N = len(data)
		p = np.zeros((N, N))

	def transform(self, data):
		raise NotImplementedException('t-SNE transform not implemented')