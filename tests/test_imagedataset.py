import os
import cv2
import pytest
import numpy as np

from tfwrapper.dataset import ImageDataset
from tfwrapper.dataset import ImageLoader
from tfwrapper.dataset import FeatureLoader
from tfwrapper.dataset import ImagePreprocessor
from tfwrapper.models.frozen import FrozenInceptionV3

from utils import curr_path
from utils import remove_dir
from utils import create_tmp_dir


def test_create_from_datastructure():
    try:
        size = 10
        root_folder = create_tmp_dir(size=size)
        dataset = ImageDataset(root_folder=root_folder)

        assert size == len(dataset)
        assert size == len(dataset.X)
        assert size == len(dataset.y)
    finally:
        remove_dir(root_folder)


def test_create_from_weird_datastructure():
    try:
        root = os.path.join(curr_path, 'test')
        folders = ['Bever','bjoern','Ekorn','Elg','Fugl','Gaupe','Grevling','Hare','Hjort','Ilder','Jerv','Katt','Maar','Mennesker - Gaaende','Mennesker - Hest','Mennesker - kjoeretoey','Mennesker - Motorsykkel','Mennesker - Syklende','Raadyr','Rein','Rev','Roeyskatt','Sau','Smaagnager','Snoemus','Ulv','Villsvin']
        os.mkdir(root)
        for folder in folders:
            dest = os.path.join(root, folder)
            os.mkdir(dest)
            img = np.ones([28, 28, 3])
            dest = os.path.join(dest, 'test.jpg')
            cv2.imwrite(dest, img)

        dataset = ImageDataset(root_folder=root)
        assert len(dataset) == len(folders)
    finally:
        remove_dir(root)


def create_tmp_labels_file(root_folder, name):
    with open(name, 'w') as f:
        for filename in os.listdir(root_folder):
            f.write('label,' + filename + '\n')

    return name


def test_create_from_labels_file():
    try:
        size = 10
        parent = create_tmp_dir(size=size)
        root_folder = os.path.join(parent, 'x')
        labels_file = os.path.join(curr_path, 'tmp.csv')
        labels_file = create_tmp_labels_file(root_folder, labels_file)

        dataset = ImageDataset(root_folder=root_folder, labels_file=labels_file)

        assert size / 2 == len(dataset.X)
        assert size / 2 == len(dataset.y)
    finally:
        remove_dir(parent)
        os.remove(labels_file)


def test_imagedataset_inheritance():
    X = np.arange(10)
    y = np.arange(10)
    labels = [str(x) for x in np.arange(10)]
    loader = ImageLoader()
    dataset = ImageDataset(X=X, y=y, labels=labels, loader=loader)
    dataset = dataset.onehot()

    assert np.array_equal(labels, dataset.labels)
    assert id(loader) == id(dataset.loader)


def test_imageloader_shape():
    try:
        size = 10
        root_folder = create_tmp_dir(size=size)
        dataset = ImageDataset(root_folder=root_folder)

        assert [size, -1, -1, 3] == dataset.shape
    finally:
        remove_dir(root_folder)


def test_resized_imageloader_shape():
    try:
        size = 10
        root_folder = create_tmp_dir(size=size)

        preprocessor = ImagePreprocessor()
        preprocessor.resize_to = (64, 64)
        loader = ImageLoader(preprocessor=preprocessor)
        dataset = ImageDataset(root_folder=root_folder, loader=loader)

        assert [size, 64, 64, 3] == dataset.shape
    finally:
        remove_dir(root_folder)


def test_set_preprocessor():
    try:
        size = 10
        root_folder = create_tmp_dir(size=size)

        dataset = ImageDataset(root_folder=root_folder)

        preprocessor = ImagePreprocessor()
        preprocessor.resize_to = (64, 64)
        dataset.preprocessor = preprocessor

        assert (10, 64, 64, 3) == dataset.X.shape, 'Setting preprocessor in dataset does not work'

        preprocessor = ImagePreprocessor()
        preprocessor.resize_to = (128, 128)
        dataset.preprocessor = preprocessor

        assert (10, 128, 128, 3) == dataset.X.shape, 'Setting preprocessor in dataset does not delete cached data'
    finally:
        remove_dir(root_folder)
