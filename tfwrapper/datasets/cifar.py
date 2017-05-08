import os
import tarfile
import numpy as np

from tfwrapper import config
from tfwrapper.utils.files import download_file

from .utils import unpickle

curr_path = config.DATASETS

def download_cifar10(size=600000, verbose=False):
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    root_folder = os.path.join(curr_path, 'cifar10')
    if not os.path.isdir(root_folder):
        os.mkdir(root_folder)

    datafolder = os.path.join(root_folder, 'cifar-10-batches-py')
    if not os.path.isdir(datafolder):

        tgz_file = os.path.join(root_folder, 'cifar-10-python.tar.gz')
        if not os.path.isfile(tgz_file):
            download_file(url, tgz_file, verbose=verbose)

        with tarfile.open(tgz_file, 'r') as f:
            if verbose:
                print('Extracting cifar10 data')
            for item in f:
                f.extract(item, root_folder)

        os.remove(tgz_file)

    metafile = os.path.join(datafolder, 'batches.meta')
    metadata = unpickle(metafile)
    print(metadata)
    label_names = [x.decode('UTF-8') for x in metadata[b'label_names']]
    print(label_names)

    X = []
    y = []

    for i in range(1, 6):
        batch = os.path.join(datafolder, 'data_batch_%d' % i)
        data = unpickle(batch)
        images = data[b'data']
        labels = data[b'labels']

        for j in range(len(images)):
            img = images[j]
            label = label_names[labels[j]]

            r = np.reshape(img[:32*32], (32, 32, 1))
            g = np.reshape(img[32*32:(32*32)*2], (32, 32, 1))
            b = np.reshape(img[(32*32)*2:], (32, 32, 1))
            rgb = np.concatenate([r, g, b], axis=2)

            X.append(rgb)
            y.append(label)

            if j % 1000 == 0 and verbose:
                print("Read %i of %i images" % (j + (10000 * i), 60000))

    return np.asarray(X), np.asarray(y)
