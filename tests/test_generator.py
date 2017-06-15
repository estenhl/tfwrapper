import pytest
import numpy as np

from tfwrapper.dataset import Dataset
from tfwrapper.dataset import ImageDataset

from utils import remove_dir
from utils import create_tmp_dir

def test_generator():
    num_items = 10
    batch_size = 2
    num_batches = int(num_items/batch_size)

    X = np.arange(num_items)
    y = np.arange(num_items)
    dataset = Dataset(X=X, y=y).shuffle()
    generator = dataset.batch_generator(batch_size)

    batches = 0
    for X, y in generator:
        assert batch_size == len(X) == len(y)
        for i in range(len(X)):
            assert X[i] == y[i]
        batches += 1

    assert num_batches == batches

def test_generator_normalization():
    X = np.arange(9)
    y = np.arange(9)
    dataset = Dataset(X=X, y=y)
    generator = dataset.batch_generator(3, normalize=True)

    for X, y in generator:
        assert 0 > X[0]
        assert 0 == X[1]
        assert 0 < X[2]
        assert X[0] == -X[2]

def test_generator_shuffling():
    np.random.seed(5)
    
    X = np.arange(10)
    y = np.arange(10)
    dataset = Dataset(X=X, y=y)
    generator = dataset.batch_generator(5, shuffle=True)

    for X, y in generator:
        for i in np.arange(5):
            assert X[i] == y[i]
        assert X[0] != 0
        break

def test_image_generator():
    try:
        num_items = 10
        batch_size = 2
        num_batches = int(num_items/batch_size)

        root_folder = create_tmp_dir(size=num_items)
        dataset = ImageDataset(root_folder=root_folder)
        generator = dataset.batch_generator(batch_size)

        batches = 0
        for X, y in generator:
            assert batch_size == len(X) == len(y)
            batches += 1

        assert num_batches == batches
    finally:
        remove_dir(root_folder)

def test_generator_slicing():
    X = np.concatenate([np.zeros(5), np.ones(5)])
    y = np.concatenate([np.zeros(5), np.ones(5)])
    dataset = Dataset(X=X, y=y)
    generator = dataset.batch_generator(5)
    zeros = generator[:5]
    ones = generator[5:]

    assert len(zeros) == len(ones) == 5

    for X, y in zeros:
        assert len(X) == len(y)
        for i in range(len(X)):
            assert X[i] == y[i] == 0

    for X, y in ones:
        assert len(X) == len(y)
        for i in range(len(X)):
            assert X[i] == y[i] == 1