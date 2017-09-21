import pytest
import numpy as np

from tfwrapper.dataset import BoundingBoxDataset


def test_translated_labels():
    imgs = np.zeros((3, 3, 3, 3))
    bboxes = np.asarray([
        [
            ['c', [1, 2, 3, 4]]
        ],
        [
            ['a', [1, 2, 3, 4]],
            ['b', [1, 2, 3, 4]]
        ],
        []
    ])

    dataset = BoundingBoxDataset(X=imgs, y=bboxes)
    dataset = dataset.translated_labels()

    expected_bboxes = np.asarray([
        [
            [3, [1, 2, 3, 4]]
        ],
        [
            [1, [1, 2, 3, 4]],
            [2, [1, 2, 3, 4]]
        ],
        []
    ])
    expected_labels = np.asarray(['background', 'a', 'b', 'c'])

    assert np.array_equal(expected_bboxes, dataset.y)
    assert np.array_equal(expected_labels, dataset.labels)


def test_translated_labels_with_paths():
    X = np.zeros((3, 3, 3, 3))
    y = []
    paths = ['a', 'b', 'c', 'd']

    dataset = BoundingBoxDataset(X=X, y=y, paths=paths)
    dataset = dataset.translated_labels()

    assert paths == dataset.paths, 'BoundingBoxDataset.translated_labels() does not preserve paths'


def test_shuffle():
    size = 5
    seed = 12345 # Ensures we are not super-unlucky and get a shuffle where everything stays in place

    X = y = paths = np.arange(size)
    dataset = BoundingBoxDataset(X=X, y=y, paths=paths)
    dataset = dataset.shuffled(12345)

    for i in range(size):
        assert (dataset.X[i] == dataset.y[i]), 'BoundingBoxDataset.shuffle mixes up images and labels'
        assert (dataset.X[i] == dataset.paths[i]), 'BoundingBoxDataset.shuffle mixes up images and paths'


def test_split():
    size = 4
    dataset_size = int(size/2)

    X = y = paths = np.arange(size)
    dataset = BoundingBoxDataset(X=X, y=y, paths=paths)
    d1, d2 = dataset.split(0.5)

    print(len(d1))
    assert len(d1) == dataset_size, 'First dataset does not get the correct len after BoundingBoxDataset.split'
    assert len(d1.X) == dataset_size, 'First dataset does not get the correct number of images after BoundingBoxDataset.split'
    assert len(d1.y) == dataset_size, 'First dataset does not get the correct number of labels after BoundingBoxDataset.split'
    assert len(d1.paths) == dataset_size, 'First dataset does not get the correct number of paths after BoundingBoxDataset.split'

    assert len(d2) == dataset_size, 'Second dataset does not get the correct len after BoundingBoxDataset.split'
    assert len(d2.X) == dataset_size, 'Second dataset does not get the correct number of images after BoundingBoxDataset.split'
    assert len(d2.y) == dataset_size, 'Second dataset does not get the correct number of labels after BoundingBoxDataset.split'
    assert len(d2.paths) == dataset_size, 'Second dataset does not get the correct number of paths after BoundingBoxDataset.split'

    for i in range(int(size/2)):
        assert d1.X[i] == d1.y[i], 'BoundingBoxDataset.split does not maintain relationship between images and labels in first dataset'
        assert d1.X[i] == d1.paths[i], 'BoundingBoxDataset.split does not maintain relationship between images and paths in first dataset'
        assert d2.X[i] == d2.y[i], 'BoundingBoxDataset.split does not maintain relationship between images and labels in second dataset'
        assert d2.X[i] == d2.paths[i], 'BoundingBoxDataset.split does not maintain relationship between images and paths in second dataset'
