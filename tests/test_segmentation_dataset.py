import pytest
import numpy as np

from tfwrapper.dataset import SegmentationDataset


def test_resized():
    X = np.asarray([np.ones((100, 50, 3)), np.ones((20, 10, 3)), np.ones((10, 5, 3))])
    y1 = np.concatenate([np.ones((50, 50)), np.zeros((50, 50))], axis=0)
    y2 = np.concatenate([np.ones((10, 10)), np.zeros((10, 10))], axis=0)
    y3 = np.concatenate([np.ones((5, 5)), np.zeros((5, 5))], axis=0)
    y = np.asarray([y1, y2, y3])

    dataset = SegmentationDataset(X=X, y=y)
    dataset = dataset.resized(max_size=(20, 10))

    assert 3 == len(dataset.X), 'Resize changes length of dataset.X array'
    assert ((20, 10, 3)) == dataset.X[0].shape, 'Segmentation dataset resize does not handle images with shape > max_size'
    assert ((20, 10, 3)) == dataset.X[1].shape, 'Segmentation dataset resize does not handle images with shape == max_size'
    assert ((20, 10, 3)) == dataset.X[2].shape, 'Segmentation dataset resize does not handle images with shape < max_size'

    expected_y = y2

    assert 3 == len(dataset.y), 'Resize changes length of dataset.y array'
    assert np.array_equal(expected_y, dataset.y[0]), 'Segmentation dataset resize does not handle labels with shape > max_size'
    assert np.array_equal(expected_y, dataset.y[1]), 'Segmentation dataset resize does not handle labels with shape == max_size'
    assert np.array_equal(expected_y, dataset.y[2]), 'Segmentation dataset resize does not handle labels with shape < max_size'


def test_squarepadded():
    X1 = np.zeros((9, 5, 3))
    y1 = np.zeros((9, 5), dtype=int)
    for i in range(9):
        for j in range(5):
            X1[i][j] = np.tile(i * j, 3)
            y1[i][j] = i * j

    X2 = np.zeros((9, 9, 3))
    y2 = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            X2[i][j] = np.tile(i * j, 3)
            y2[i][j] = i * j

    X3 = np.zeros((5, 9, 3))
    y3 = np.zeros((5, 9), dtype=int)
    for i in range(5):
        for j in range(9):
            X3[i][j] = np.tile(i * j, 3)
            y3[i][j] = i * j

    X = np.asarray([X1, X2, X3])
    y = np.asarray([y1, y2, y3])

    dataset = SegmentationDataset(X=X, y=y)
    dataset = dataset.squarepadded()

    assert (3, 9, 9, 3) == dataset.X.shape
    assert (3, 9, 9) == dataset.y.shape


def test_squarepadded_odd():
    X1 = np.zeros((10, 5, 3))
    y1 = np.zeros((10, 5), dtype=int)
    for i in range(10):
        for j in range(5):
            X1[i][j] = np.tile(i * j, 3)
            y1[i][j] = i * j

    X2 = np.zeros((10, 10, 3))
    y2 = np.zeros((10, 10), dtype=int)
    for i in range(10):
        for j in range(10):
            X2[i][j] = np.tile(i * j, 3)
            y2[i][j] = i * j

    X3 = np.zeros((5, 10, 3))
    y3 = np.zeros((5, 10), dtype=int)
    for i in range(5):
        for j in range(10):
            X3[i][j] = np.tile(i * j, 3)
            y3[i][j] = i * j

    X = np.asarray([X1, X2, X3])
    y = np.asarray([y1, y2, y3])

    dataset = SegmentationDataset(X=X, y=y)
    dataset = dataset.squarepadded()

    assert (3, 10, 10, 3) == dataset.X.shape
    assert (3, 10, 10) == dataset.y.shape


def test_framed_X():
    X = np.asarray([np.ones((10, 10, 3))] * 3)
    y = np.asarray([np.ones((10, 10))] * 3)

    print('X.shape: ' + str(X.shape))
    dataset = SegmentationDataset(X=X, y=y)
    dataset = dataset.framed_X((4, 6))

    assert (3, 14, 16, 3) == dataset.X.shape
    assert (3, 10, 10) == dataset.y.shape


def test_framed_X_odd():
    X = np.asarray([np.ones((10, 10, 3))] * 3)
    y = np.asarray([np.ones((10, 10))] * 3)

    dataset = SegmentationDataset(X=X, y=y)
    dataset = dataset.framed_X((5, 7))

    assert (3, 15, 17, 3) == dataset.X.shape
    assert (3, 10, 10) == dataset.y.shape