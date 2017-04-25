import os
import cv2
import pytest
import numpy as np

from tfwrapper import ImageLoader
from tfwrapper import ImageDataset

from utils import curr_path
from utils import remove_dir

def create_tmp_dir(root=os.path.join(curr_path, 'tmp'), size=10):
    os.mkdir(root)
    for label in ['x', 'y']:
        os.mkdir(os.path.join(root, label))
        for i in range(int(size/2)):
            img = np.zeros((10, 10, 3))
            path = os.path.join(root, label, str(i) + '.jpg')
            cv2.imwrite(path, img)

    return root

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