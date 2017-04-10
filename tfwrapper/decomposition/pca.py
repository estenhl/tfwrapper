import numpy as np
from scipy import linalg

def pca(points, dimensions=2):
	cov = np.cov(points, rowvar=False)
	eigenvalues, eigenvectors = linalg.eigh(points)
	idx = np.argsort(eigenvalues)[::-1]