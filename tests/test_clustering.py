import pytest
import numpy as np

from tfwrapper.clustering import kmeans
from tfwrapper.clustering import KMEANS_INIT_RANDOM_PARTITION as randpar

def test_kmeans():
    points = np.asarray([[1, 1], [2, 2], [3, 3], [4, 4]])
    num_clusters = 2

    clusters = kmeans(num_clusters, points)

    assert num_clusters == len(clusters), 'kmeans does not return correct number of clusters'
    for elem in clusters[0]:
        assert not elem in clusters[1], 'kmeans returns overlapping clusters'


def test_kmeans_with_random_partitioning_init():
    points = np.asarray([[1, 1], [2, 2], [3, 3], [4, 4]])
    num_clusters = 2

    clusters = kmeans(num_clusters, points, init=randpar)

    assert num_clusters == len(clusters), 'kmeans with init=\'random_partition\' does not return correct number of clusters'
    for elem in clusters[0]:
        assert not elem in clusters[1], 'kmeans with init=\'random_partition\' returns overlapping clusters'


def test_kmeans_with_invalid_k():
    points = np.asarray([[1, 1], [2, 2], [3, 3], [4, 4]])
    num_clusters = 5

    exception = False
    try:
        clusters = kmeans(num_clusters, points)
    except Exception:
        exception = True

    assert exception, 'kmeans does not throw an exception when k > num_points'
