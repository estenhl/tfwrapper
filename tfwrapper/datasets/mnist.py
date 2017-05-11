import os
import cv2
import gzip
import numpy as np
from struct import unpack

from tfwrapper import logger
from tfwrapper.utils.files import download_file

from .utils import setup_structure
from .utils import recursive_delete

def download_mnist():
    data_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'

    root_path, data_path, labels_file = setup_structure('mnist')
    local_data_zip = os.path.join(root_path, 'data.zip')
    local_labels_zip = os.path.join(root_path, 'labels.zip')

    num_files = len(os.listdir(data_path))
    if True: #num_files < zzz
        if not os.path.isfile(local_data_zip):
            download_file(data_url, local_data_zip)
        if not os.path.isfile(local_labels_zip):
            download_file(labels_url, local_labels_zip)

    return local_data_zip, local_labels_zip

def parse_mnist(size=None, imagesize=[28, 28]):
    logger.info('Unpacking mnist data with size %s' % imagesize)

    data_file, labels_file = download_mnist()
    images = gzip.open(data_file, 'rb')
    labels = gzip.open(labels_file, 'rb')
    height, width = imagesize

    images.read(4)
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]
    if size is not None:
        N = size

    X = np.zeros((N, rows, cols), dtype=np.float32)
    resized_X = np.zeros((N, height, width))
    y = np.zeros((N, 1), dtype=np.uint8)
    for i in range(N):
        if i % 1000 == 0:
            logger.info("Read %i of %i images" % (i, N))
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                X[i][row][col] = tmp_pixel
        resized_X[i] = cv2.resize(X[i], (height, width))
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    logger.info('Read %i images' % N)

    resized_X = np.expand_dims(resized_X, axis=3)

    return resized_X, y
