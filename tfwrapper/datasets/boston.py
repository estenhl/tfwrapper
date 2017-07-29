import os
import numpy as np
import pandas as pd

from tfwrapper import config
from tfwrapper.utils.files import download_file

# See the file {TFWRAPPER_LOCATION}/data/datasets/boston/housing.names for info
headers = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


DEFAULT_HEADER_INDEX = 13


def parse_boston(y_index=DEFAULT_HEADER_INDEX):
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
    readme_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names'
    path = os.path.join(config.DATASETS, 'boston')
    data_file = os.path.join(path, 'housing.data')
    readme_file = os.path.join(path, 'housing.names')

    if not os.path.isdir(path):
        os.mkdir(path)

    if not os.path.isfile(data_file):
        download_file(data_url, data_file)

    if not os.path.isfile(readme_file):
        download_file(readme_url, readme_file)

    if y_index >= len(headers):
        logger.error('Invalid y_index %d. (Valid is < %d)' % (y_index, len(headers)))
        logger.error('Defaulting to %d' % DEFAULT_HEADER_INDEX)
        y_index = DEFAULT_HEADER_INDEX

    y_header = headers[y_index]
    X_headers = headers.copy()
    del X_headers[y_index]

    df = pd.read_csv(data_file, delim_whitespace=True, header=None, names=headers)
    X = np.asarray(df[X_headers]).astype(np.float32)
    y = np.asarray(df[[y_header]]).astype(np.float32)

    return X, y

