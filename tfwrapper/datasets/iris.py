import os
import numpy as np

from tfwrapper import config
from tfwrapper.utils.files import download_from_google_drive


def parse_iris():
    gdrive_id = '0B7gC90mSfjC6alRmU1NUUU1ibzQ'
    folder = os.path.join(config.DATASETS, 'iris')
    path = os.path.join(folder, 'iris.csv')

    if not os.path.isdir(folder):
        os.mkdir(folder)

    if not os.path.isfile(path):
        download_from_google_drive(gdrive_id, path)

    with open(path, 'r') as f:
        X = []
        y = []
        for line in f.readlines()[1:]:
            tokens = line.strip().split(',')
            X.append(tokens[:-1])
            y.append(tokens[-1])

    return np.asarray(X), np.asarray(y)