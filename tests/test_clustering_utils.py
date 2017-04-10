import pytest
import numpy as np

from tfwrapper.clustering.utils import find_centroid
from tfwrapper.clustering.utils import cluster_points

def test_find_centroid_2d():
	points = np.asarray([
		[10, 0],
		[0, 10],
		[-10, 0],
		[0, -10]
	])

	actual_centroid = np.asarray([0, 0])
	centroid = find_centroid(points)

	assert np.array_equal(actual_centroid, centroid)

def test_find_centroid_3d():
	points = np.asarray([
		[10, 0, 10],
		[10, 0, 0],
		[10, 0, -10],
		[0, 10, 10],
		[0, 10, 0],
		[0, 10, -10],
		[-10, 0, 10],
		[-10, 0, 0],
		[-10, 0, -10],
		[0, -10, 10],
		[0, -10, 0],
		[0, -10, -10]
	])

	actual_centroid = np.asarray([0, 0, 0])
	centroid = find_centroid(points)
	
	assert np.array_equal(actual_centroid, centroid)

def contains(arr, point):
	for elem in arr:
		if np.array_equal(elem, point):
			return True

	return False

def test_cluster_points_2d():
	points = np.asarray([
		[10, 9],
		[10, 11],
		[9, 10],
		[11, 10],
		[100, 99],
		[100, 101],
		[99, 100],
		[101, 100]
	])

	centroids = np.asarray([
		[10, 10],
		[100, 100]
	])

	clusters = cluster_points(points, centroids)
	assert len(clusters) == 2
	assert len(clusters[0]) == 4
	assert len(clusters[1]) == 4

	for point in points[:4]:
		assert contains(clusters[0], point)
	for point in points[4:]:
		assert contains(clusters[1], point)

def test_cluster_points_3d():
	points = np.asarray([
		[10, 9, 10],
		[10, 11, 10],
		[9, 10, 10],
		[11, 10, 10],
		[100, 99, 100],
		[100, 101, 100],
		[99, 100, 100],
		[101, 100, 100]
	])

	centroids = np.asarray([
		[10, 10, 10],
		[100, 100, 100]
	])

	clusters = cluster_points(points, centroids)
	assert len(clusters) == 2
	assert len(clusters[0]) == 4
	assert len(clusters[1]) == 4

	for point in points[:4]:
		assert contains(clusters[0], point)
	for point in points[4:]:
		assert contains(clusters[1], point)
