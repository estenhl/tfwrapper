import os
import logging
import tarfile
import numpy as np

from tfwrapper import config
from tfwrapper import logger
from tfwrapper.utils.files import download_file

from .utils import unpickle

curr_path = config.DATASETS

def download_cifar(name, datafolder_name):
    url_name = '%s-%s-python' % (name[:5], name[5:])
    url = 'https://www.cs.toronto.edu/~kriz/%s.tar.gz' % url_name

    root_folder = os.path.join(curr_path, name)
    if not os.path.isdir(root_folder):
        os.mkdir(root_folder)

    datafolder = os.path.join(root_folder, datafolder_name)
    if not os.path.isdir(datafolder):

        tgz_file = os.path.join(root_folder, '%s.tar.gz' % url_name)
        if not os.path.isfile(tgz_file):
            download_file(url, tgz_file)

        with tarfile.open(tgz_file, 'r') as f:
            logger.info('Extracting %s data' % name)
            for item in f:
                f.extract(item, root_folder)

        os.remove(tgz_file)

    return datafolder

def parse_cifar_image(img):
    r = np.reshape(img[:32*32], (32, 32, 1))
    g = np.reshape(img[32*32:(32*32)*2], (32, 32, 1))
    b = np.reshape(img[(32*32)*2:], (32, 32, 1))
    rgb = np.concatenate([r, g, b], axis=2)

    return rgb

def parse_cifar10(size=50000):
    datafolder = download_cifar('cifar10', 'cifar-10-batches-py')
    metafile = os.path.join(datafolder, 'batches.meta')
    metadata = unpickle(metafile)
    label_names = [x.decode('UTF-8') for x in metadata[b'label_names']]

    X = []
    y = []

    for i in range(0, 5):
        if len(X) >= size:
            break

        batch = os.path.join(datafolder, 'data_batch_%d' % (i + 1))
        data = unpickle(batch)
        images = data[b'data']
        labels = data[b'labels']

        for j in range(len(images)):
            img = parse_cifar_image(images[j])
            label = label_names[labels[j]]

            X.append(img)
            y.append(label)

            if len(X) >= size:
                break

            if j % 1000 == 0:
                logger.info("Read %i of %i cifar10 images" % ((10000 * i) + j, size))

    logger.info("Read %i images" % len(X))

    return np.asarray(X), np.asarray(y)

def parse_cifar10_test(size=10000):
    datafolder = download_cifar('cifar10', 'cifar-10-batches-py')
    metafile = os.path.join(datafolder, 'batches.meta')
    metadata = unpickle(metafile)
    label_names = [x.decode('UTF-8') for x in metadata[b'label_names']]

    X = []
    y = []

    test_file = os.path.join(datafolder, 'test_batch')
    data = unpickle(test_file)
    images = data[b'data']
    labels = data[b'labels']

    for j in range(len(images)):
        img = parse_cifar_image(images[j])
        label = label_names[labels[j]]

        X.append(img)
        y.append(label)

        if len(X) >= size:
            break

        if j % 1000 == 0:
            logger.info("Read %i of %i cifar10 test images" % (j, size))

    logger.info("Read %i images" % len(X))

    return np.asarray(X), np.asarray(y)

def parse_cifar100(size=500000):
    datafolder = download_cifar('cifar100', 'cifar-100-python')

    metafile = os.path.join(datafolder, 'meta')
    metadata = unpickle(metafile)
    label_names = [x.decode('UTF-8') for x in metadata[b'fine_label_names']]

    train_file = os.path.join(datafolder, 'train')
    train = unpickle(train_file)
    images = train[b'data']
    labels = train[b'fine_labels']

    X = []
    y = []

    for i in range(min(size, len(images))):
        img = parse_cifar_image(images[i])
        label = label_names[labels[i]]

        X.append(img)
        y.append(label)

        if i % 1000 == 0:
            logger.info("Read %i of %i cifar100 images" % (i, size))

    logger.info("Read %i cifar100 images" % len(X))

    return np.asarray(X), np.asarray(y)
