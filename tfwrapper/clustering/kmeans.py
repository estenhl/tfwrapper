import random
import numpy as np

from tfwrapper.utils.exceptions import raise_exception
from tfwrapper.utils.exceptions import InvalidArgumentException

from .utils import find_centroid
from .utils import cluster_points

INIT_FORGY = 'forgy'
INIT_RANDOM_PARTITION = 'random_partition'

def _init_forgy(k, points):
    idx = np.arange(len(points))
    np.random.shuffle(idx)
    idx = idx[:k]

    return points[idx]


def _init_random_partition(k, points):
    idx = np.arange(len(points))
    shuffled = points[idx]
    clusters = np.array_split(shuffled, k)

    return [find_centroid(cluster) for cluster in clusters]


def kmeans(k, points, dimensions=None, epochs=500, init=INIT_FORGY):
    if k > len(points):
        raise_exception('Unable to do kmeans when num_clusters (%d) > num_points (%d)' % (k, len(points)), InvalidArgumentException)

    if dimensions is None:
        dimensions = points.shape[-1]

    if init == INIT_FORGY:
        centroids = _init_forgy(k, points)
    elif init == INIT_RANDOM_PARTITION:
        centroids = _init_random_partition(k, points)
    else:
        raise_exception('Invalid initialization schema %s. (Valid is %s)' % (init, str([INIT_FORGY, INIT_RANDOM_PARTITION])), InvalidArgumentException)
    
    while True:
        prev_centroids = centroids.copy()
        clusters = cluster_points(points, centroids)
        for j in range(len(clusters)):
            centroids[j] = find_centroid(clusters[j])

        if np.array_equal(centroids, prev_centroids):
            return clusters





    