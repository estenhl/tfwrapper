import numpy as np
import matplotlib.pyplot as plt

from tfwrapper.metrics.distance import euclidean

colours = ['red', 'green', 'blue', 'yellow', 'purple', 'brown', 'teal', 'orange', 'pink', 'gray', 'darkslategray', 'whitesmoke', 'gold', 'tan', 'midnightblue', 'thistle', 'moccasin']

def plot_boundaries(width, height, centroids):
	border_x = []
	border_y = []
	x_step = width / 500
	y_step = height / 500
	i = 0
	while i < height:
		j = 0
		while j < width:
			point = np.asarray([i, j])
			distances = [euclidean(point, centroid) for centroid in centroids]
			distances = sorted(distances)
			if distances[1] - distances[0] < max(x_step, y_step):
				border_x.append(i)
				border_y.append(j)
			j += x_step
		i += y_step

	plt.scatter(border_x, border_y, color='black', s=0.1)

def plot_clusters(clusters, centroids=None, names=None, title='Clusters', figsize=(10, 10), height=None, width=None, plot_decision_boundaries=False):
	fig = plt.figure(figsize=figsize)

	if height:
		plt.gca().set_ylim([0, height])
	if width:
		plt.gca().set_xlim([0, width])

	handles = []
	for i in range(len(clusters)):
		handle = plt.scatter(clusters[i][:,0], clusters[i][:,1], color=colours[i%len(colours)])
		handles.append(handle)

	if centroids is not None:
		plt.scatter(centroids[:,0], centroids[:,1], marker='x', color='black')

	if names is not None and len(names) == len(handles):
		plt.legend(handles, names)

	# TODO: Height and width gots to go
	if height and width and plot_decision_boundaries:
		plot_boundaries(width, height, centroids)

	fig.suptitle(title, fontsize=30)

	plt.show()