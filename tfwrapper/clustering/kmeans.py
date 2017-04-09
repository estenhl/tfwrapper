import random
import numpy as np

from tfwrapper.metrics.distance import euclidean

def cluster_points(points, centroids, metric=euclidean):
	clusters = [[] for i in range(len(centroids))]

	for point in points:
		cluster = np.argmin([metric(point, centroid) for centroid in centroids])
		clusters[cluster].append(point)

	return [np.asarray(cluster) for cluster in clusters]

def find_centroid(points):
	return np.asarray([np.mean(points[:,0]), np.mean(points[:,1])])

def kmeans(k, points, epochs=500):
	miny = np.amin(points[:,0])
	maxy = np.amax(points[:,0])
	minx = np.amin(points[:,1])
	maxx = np.amax(points[:,1])

	centroids = np.zeros((k, 2))
	for i in range(k):
		centroids[i] = np.asarray([random.uniform(miny, maxy), random.uniform(minx, maxx)])

	for i in range(epochs):
		clusters = cluster_points(points, centroids)
		for j in range(len(clusters)):
			centroid = find_centroid(clusters[j])
			centroids[j] = find_centroid(clusters[j])

	return clusters