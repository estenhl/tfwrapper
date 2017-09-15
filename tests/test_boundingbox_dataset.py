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
    expected_labels = np.asarray(['a', 'b', 'c'])

    assert np.array_equal(expected_bboxes, dataset.y)
    assert np.array_equal(expected_labels, dataset.labels)


def test_translated_labels_with_paths():
    X = np.zeros((3, 3, 3, 3))
    y = []
    paths = ['a', 'b', 'c', 'd']

    dataset = BoundingBoxDataset(X=X, y=y, paths=paths)
    dataset = dataset.translated_labels()

    assert paths == dataset.paths, 'BoundingBoxDataset.translated_labels() does not preserve paths'