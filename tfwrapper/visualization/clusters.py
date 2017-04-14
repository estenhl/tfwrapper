import matplotlib.pyplot as plt

colours = ['red', 'green', 'blue', 'yellow', 'purple', 'brown', 'teal', 'orange', 'pink']

def plot_clusters(clusters, centroids=None, names=None, title='Clusters', figsize=(10, 10), height=None, width=None):
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

	fig.suptitle(title, fontsize=30)

	plt.show()