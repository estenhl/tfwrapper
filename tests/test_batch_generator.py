import pytest
import numpy as np

from tfwrapper import Dataset

def test_generator():
    size = 100
    batch_size = 10

    X = np.arange(size)
    y = np.arange(size)
    dataset = Dataset(X=X, y=y)

    num_batches = 0
    for X, y in dataset.batch_generator(batch_size):
        assert len(X) == batch_size
        assert len(y) == batch_size
        for i in range(batch_size):
            assert (num_batches * batch_size) + i in X
            assert (num_batches * batch_size) + i in y
            assert X[i] == y[i]
        num_batches += 1

    assert size / batch_size == num_batches

def test_uneven_generator():
    size = 95
    batch_size = 10

    X = np.arange(size)
    y = np.arange(size)
    dataset = Dataset(X=X, y=y)

    num_batches = 0
    for X, y in dataset.batch_generator(batch_size):
        num_batches += 1

    assert size % batch_size == len(X)
    assert size % batch_size == len(y)
    assert int(size / batch_size) + 1 == num_batches

def test_generator_normalize():
    X = np.arange(1, 10)
    y = np.arange(1, 10)
    dataset = Dataset(X=X, y=y)

    num_batches = 0
    for X, y in dataset.batch_generator(3, normalize=True):
        assert np.mean(X) == 0

def test_generator_shuffle():
    batch_size = 50

    X = np.arange(100)
    y = np.arange(100)
    dataset = Dataset(X=X, y=y)

    num_batches = 0
    np.random.seed(4)
    for X, y in dataset.batch_generator(batch_size, shuffle=True):
        for i in range(batch_size):
            assert X[i] == y[i]
            assert (num_batches * batch_size) + i in X
        assert X[0] != num_batches * batch_size
        num_batches += 1

def test_infinite_generator():
    size = 10
    batch_size = 5
    total_batches = int(size / batch_size)

    X = np.arange(size)
    y = np.arange(size)
    dataset = Dataset(X=X, y=y)

    exception = False
    num_batches = 0
    try:
        for X, y in dataset.batch_generator(batch_size, infinite=True):
            num_batches += 1
            if num_batches > total_batches * 2:
                break
    except Exception:
        exception = True

    assert exception, 'Infinite (and scary) generators does not throw an exception'

