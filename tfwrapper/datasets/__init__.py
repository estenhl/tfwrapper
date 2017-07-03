import os
import cv2
import gzip
import math
import tarfile
import zipfile
import urllib.request
import numpy as np
from shutil import copyfile
from struct import unpack

from tfwrapper import config
from tfwrapper import logger
from tfwrapper import Dataset
from tfwrapper import ImageDataset
from tfwrapper.utils.files import download_file

from .iris import parse_iris
from .wine import headers as wine_headers
from .wine import download_wine
from .utils import setup_structure
from .utils import recursive_delete
from .utils import curr_path
from .mnist import parse_mnist
from .cifar import parse_cifar10
from .cifar import parse_cifar10_test
from .cifar import parse_cifar100
from .boston import parse_boston
from .boston import headers as boston_headers
from .catsdogs import download_cats_and_dogs
from .imagenet import parse_imagenet_labels

curr_path = config.DATASETS

def download_ptb():
    url = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
    root_path, _, labels_file = setup_structure('penn_tree_bank', create_data_folder=False)
    local_tar = os.path.join(root_path, 'simple-examples.tgz')

    examples_path = os.path.join(root_path, 'simple-examples')
    data_path = os.path.join(examples_path, 'data')
    data_file = os.path.join(data_path, 'ptb.train.txt')
    if not (os.path.isdir(examples_path) and os.path.isdir(data_path) and os.path.isfile(data_file)):
        if not os.path.isfile(local_tar):
            download_file(url, local_tar)

        with tarfile.open(local_tar, 'r') as f:
            logger.info('Extracting penn_tree_bank data')
            for item in f:
                f.extract(item, root_path)

        for folder in os.listdir(root_path):
            recursive_delete(os.path.join(root_path, folder), skip=[data_file])

    return data_file

def download_flowers():
    url = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz'

    root_path, data_folder, labels_file = setup_structure('flowers')

    if not len(os.listdir(data_folder)) > 1000:
        tgz_file = os.path.join(root_path, '17flowers.tgz')
        if not os.path.isfile(tgz_file):
            download_file(url, tgz_file)

        with tarfile.open(tgz_file, 'r') as f:
            logger.info('Extracting flowers data')
            for item in f:
                f.extract(item, root_path)

        tmp_folder = os.path.join(root_path, 'jpg')
        for filename in os.listdir(tmp_folder):
            src = os.path.join(tmp_folder, filename)

            if filename.lower().endswith('.jpg'):
                dest = os.path.join(data_folder, filename)
                copyfile(src, dest)

            os.remove(src)
        os.rmdir(tmp_folder)
        os.remove(tgz_file)

    labels = ['Buttercup', 'Colt\'s Foot', 'Daffodil', 'Daisy', 'Dandelion', 'Fritillary', 'Iris', 'Pansy', 'Sunflower', 'Windflower', 'Snowdrop', 'Lily Valley', 'Bluebell', 'Crocus', 'Tigerlily', 'Tulip', 'Cowslip']
    if not os.path.isfile(labels_file):
        with open(labels_file, 'w') as f:
            for i in range(0, 17):
                for j in range(1, 80):
                    index = str((i * 80) + j)
                    while len(index) < 4:
                        index = '0' + index
                    f.write('%s,image_%s.jpg\r\n' % (labels[i], index))

    return data_folder, labels_file

def cats_and_dogs(size=25000):
    if size is not 25000:
        logger.warning('Size not implemented for cats and dogs dataset')
    data_path = download_cats_and_dogs()
    dataset = ImageDataset(root_folder=data_path)

    return dataset

def mnist(size=None, imagesize=[28, 28]):
    X, y = parse_mnist(size=size, imagesize=imagesize)
    dataset = Dataset(X=X, y=np.asarray(y).flatten())
    return dataset

def flowers(size=1360):
    data_path, labels_file = download_flowers()

    if size < 1360:
        tmp_labels_file = os.path.join(os.path.dirname(labels_file), 'tmp.txt')
        with open(labels_file, 'r') as f:
            lines = f.readlines()

        num_classes = 17
        num_flowers_per_class = int(math.floor(size / num_classes))

        class_counts = np.repeat(num_flowers_per_class, num_classes)
        for i in range(size - num_flowers_per_class * num_classes):
            class_counts[i] += 1

        total_flowers_per_class = 80
        with open(tmp_labels_file, 'w') as f:
            for i in range(num_classes):
                for j in range(class_counts[i]):
                    f.write(lines[(i * total_flowers_per_class) + j])

        dataset = ImageDataset(root_folder=data_path, labels_file=tmp_labels_file)
        os.remove(tmp_labels_file)
    else:
        dataset = ImageDataset(root_folder=data_path, labels_file=labels_file)
    
    return dataset

def cifar10(size=50000, test=False, include_test=False):
    if include_test:
        X, y = parse_cifar10(size=size)
        test_X, test_y = parse_cifar10_test()
        return Dataset(X=X, y=y), Dataset(X=test_X, y=test_y)
    elif test:
        X, y =parse_cifar10_test(size=size)
    else:
        X, y = parse_cifar10(size=size)

    return Dataset(X=X, y=y)

def cifar100(size=50000):
    X, y = parse_cifar100(size=size)

    return Dataset(X=X, y=y)

def wine(y=None, include_headers=False, size=178):
    X, y = download_wine(y_index=y, size=size)
    dataset = Dataset(X=np.asarray(X), y=np.reshape(np.asarray(y), (len(y), 1)))

    if include_headers:
        return dataset, wine_headers
    
    return dataset

def imagenet(include_labels=False):
    logger.error('Download of imagenet is not implemented')

    if include_labels:
        return None, parse_imagenet_labels()

    return None

def boston(include_headers=False, y_index=13):
    X, y = parse_boston(y_index=y_index)
    dataset = Dataset(X=X, y=y)

    if include_headers:
        return dataset, boston_headers

    return dataset

def iris():
    X, y = parse_iris()

    return Dataset(X=X, y=y)
