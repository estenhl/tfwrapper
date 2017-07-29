import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def plot_clusters(clusters, centroids=None, plot_decision_boundaries=False, dimensions=2, **pltargs):
    if 'figsize' in pltargs:
        fig = plt.figure(figsize=pltargs['figsize'])
    else:
        fig = plt.figure()

    if dimensions == 2:
        ax = fig.add_subplot(111)
    elif dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')

    if 'height' in pltargs:
        ax.set_ylim([0, pltargs['height']])
    if 'width' in pltargs:
        ax.set_xlim([0, pltargs['width']])

    handles = []
    for i in range(len(clusters)):
        if dimensions == 2:
            handle = ax.scatter(clusters[i][:,0], clusters[i][:,1], color=colours[i%len(colours)])
        elif dimensions == 3:
            handle = ax.scatter(clusters[i][:,0], clusters[i][:,1], clusters[i][:,2], color=colours[i%len(colours)])
        handles.append(handle)

    if centroids is not None:
        if dimensions == 2:
            ax.scatter(centroids[:,0], centroids[:,1], marker='x', color='black')
        elif dimensions == 3:
            ax.scatter(centroids[:,0], centroids[:,1], centroids[:, 2], marker='x', color='black')

    if 'names' in pltargs:
        ax.legend(handles, pltargs['names'])

    # TODO: Height and width gots to go
    if plot_decision_boundaries:
        raise NotImplementedError('Plotting decision boundaries is not implemented')

    if 'title' in pltargs:
        fig.suptitle(pltargs['title'], fontsize=30)

    plt.show()