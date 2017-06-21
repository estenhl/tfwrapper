import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tfwrapper.datasets import flowers
from tfwrapper.clustering import kmeans
from tfwrapper.visualization import plot_clusters


num_clusters = 5
num_points_per_cluster = 100
height = 100
width = 100


def generate_data(num_clusters, num_points_per_cluster, height, width):
	centroids = []
	points = []

	for i in range(num_clusters):
		centroid = [random.randint(height * 1/4, height * 3/4), random.randint(height * 1/4, height * 3/4)]
		y = np.random.normal(centroid[0], height/33, num_points_per_cluster)
		x = np.random.normal(centroid[1], width/33, num_points_per_cluster)
		points += list(zip(y, x))
		centroids.append(centroid)

	return np.asarray(centroids), np.asarray(points)


centroids, points = generate_data(num_clusters, num_points_per_cluster, height, width)
clusters = kmeans(num_clusters, points)
names = ['Cluster %d' % i for i in range(num_clusters)]
plot_clusters(clusters, centroids=centroids, names=names, height=height, width=width, title='KMeans example', figsize=(10, 10))

