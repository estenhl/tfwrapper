import pytest
import numpy as np

from tfwrapper import Dataset
from tfwrapper import ImageLoader
from tfwrapper import ImageDataset
from tfwrapper import ImagePreprocessor

from utils import remove_dir
from utils import create_tmp_dir

def test_generator():
    size = 100
    batch_size = 10

    X = np.arange(size)
    y = np.arange(size)
    dataset = Dataset(X=X, y=y)

    num_batches = 0
    for X, y in dataset.batch_generator(batch_size):
        assert len(X) == batch_size, 'X in yielded batches does not have the given size'
        assert len(y) == batch_size, 'y in yielded batches does not have the given size'
        for i in range(batch_size):
            assert (num_batches * batch_size) + i in X, 'Non-shuffled batches does not maintain element order for X'
            assert (num_batches * batch_size) + i in y, 'Non-shuffled batches does not maintain element order for y'
            assert X[i] == y[i], 'Non-shuffled batches does not maintain X/y relationship'
        num_batches += 1

    assert size / batch_size == num_batches, 'Incorrect number of batches'

def test_uneven_generator():
    size = 95
    batch_size = 10

    X = np.arange(size)
    y = np.arange(size)
    dataset = Dataset(X=X, y=y)

    num_batches = 0
    for X, y in dataset.batch_generator(batch_size):
        num_batches += 1

    assert size % batch_size == len(X), 'X in last batch does not contain the correct number of elements when size %% batch_size != 0'
    assert size % batch_size == len(y), 'y in last batch does not contain the correct number of elements when size %% batch_size != 0'
    assert int(size / batch_size) + 1 == num_batches, 'Number of batches are not correct when size %% batch_size != 0'

def test_generator_normalize():
    X = np.arange(1, 10)
    y = np.arange(1, 10)
    dataset = Dataset(X=X, y=y)

    num_batches = 0
    for X, y in dataset.batch_generator(3, normalize=True):
        assert np.mean(X) == 0, 'Batches are not normalized'

def test_generator_shuffle():
    batch_size = 50

    X = np.arange(100)
    y = np.arange(100)
    dataset = Dataset(X=X, y=y)

    num_batches = 0
    np.random.seed(4)
    for X, y in dataset.batch_generator(batch_size, shuffle=True):
        for i in range(batch_size):
            assert X[i] == y[i], 'Shuffling a batch screws with X/y relationship'
            assert (num_batches * batch_size) + i in X, 'Shuffling happens between batches'
        assert X[0] != num_batches * batch_size, 'Batches are not shuffled internally'
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

def test_imagedataset_generator():
    try:
        size = 10
        img_shape = (20, 20, 3)
        batch_size = 2
        root_folder = create_tmp_dir(size=size, img_shape=img_shape)
        dataset = ImageDataset(root_folder=root_folder)

        num_batches = 0
        for X, _ in dataset.batch_generator(batch_size):
            assert batch_size == len(X), 'Image dataset generator does not return batches with given size'
            assert np.ndarray == type(X[0]), 'Image dataset generator does not return images'
            assert img_shape == X[0].shape, 'Image dataset generator does not return images with correct measurements'
            
            num_batches += 1
        
        assert int(size / batch_size) == num_batches, 'Image dataset generator does not return correct number of batches'
    finally:
        remove_dir(root_folder)

def test_imagedataset_generator_with_preprocessor():
    try:
        size = 10
        batch_size = 2
        root_folder = create_tmp_dir(size=size)
        dataset = ImageDataset(root_folder=root_folder)

        img_shape = (60, 60, 3)
        preprocessor = ImagePreprocessor()
        preprocessor.resize_to = (img_shape[0], img_shape[1])
        dataset.loader = ImageLoader(preprocessor=preprocessor)

        num_batches = 0
        for X, _ in dataset.batch_generator(batch_size):
            assert batch_size == len(X), 'Image dataset generator does not return batches with given size'
            assert np.ndarray == type(X[0]), 'Image dataset generator does not return images'
            assert img_shape == X[0].shape, 'Image dataset generator does not return images with correct measurements'
            
            num_batches += 1
        
        assert int(size / batch_size) == num_batches, 'Image dataset generator does not return correct number of batches'
    finally:
        remove_dir(root_folder)