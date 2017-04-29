import random
import numpy as np

from .utils import find_centroid
from .utils import cluster_points

def kmeans(k, points, epochs=500):
	dims = points.shape[-1]

	# Generates random start centroids
	centroids = np.zeros((k, dims))
	for i in range(k):
		for j in range(dims):
			centroids[i][j] = random.uniform(np.amin(points[:,j]), np.amax(points[:,j]))

	for i in range(epochs):
		clusters = cluster_points(points, centroids)
		for j in range(len(clusters)):
			centroid = find_centroid(clusters[j])
			centroids[j] = find_centroid(clusters[j])

	return clusters