import numpy as np

from tfwrapper.metrics.distance import euclidean


def cluster_points(points, centroids, metric=euclidean):
    clusters = [[] for i in range(len(centroids))]

    for point in points:
        cluster = np.argmin([metric(point, centroid) for centroid in centroids])
        clusters[cluster].append(point)

    return [np.asarray(cluster) for cluster in clusters]


def find_centroid(points, dims=None):
    if not dims:
        dims = points.shape[-1]
    
    centroid = [np.mean(points[:,i]) for i in range(dims)]
    return centroid