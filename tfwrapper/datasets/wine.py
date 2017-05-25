import os

from tfwrapper import config
from tfwrapper import logger
from tfwrapper.utils.files import download_file
from tfwrapper.utils.exceptions import InvalidArgumentException

headers = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

def download_wine(y_index=None, size=178):
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
    path = os.path.join(config.DATASETS, 'wine')
    data_file = os.path.join(path, 'wine.data')

    if not os.path.isdir(path):
        os.mkdir(path)

    if not os.path.isfile(data_file):
        download_file(url, data_file)

    if y_index is None:
        logger.warning('Retrieving wine dataset without specifying an index for y-values')
    elif type(y_index) is int:
        if y_index > len(headers):
            errormsg = 'Invalid y_index %d: Only %d variables in wine dataset' % (y_index, len(headers))
            logger.error(errormsg)
            raise InvalidArgumentException(errormsg)
    elif type(y_index) is str:
        if y_index in headers:
            y_index = headers.index(y_index)
        else:
            errormsg = 'Invalid y_index %s: No such header in wine dataset' % y_index
            logger.error(errormsg)
            raise InvalidArgumentException(errormsg)
    else:
        errormsg = 'Invalid y_index type %s (Valid are [\'int\', \'str\'])' % type(y_index)
        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)

    X = []
    y = []

    i = 0
    for line in open(data_file, 'r'):
        data = line.split(',')
        if y_index is not None:
            y.append(float(data[y_index]))
            X.append([float(x) for x in data[:y_index] + data[y_index + 1:]])
        else:
            X.append([float(x) for x in data])

        if i == size:
            break

    logger.info('Read %i wine instances' % len(X))

    if len(y) is None:
        y = None

    return X, y

