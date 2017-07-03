import os
import numpy as np
from collections import Counter

from tfwrapper import logger
from tfwrapper.utils.files import parse_features
from tfwrapper.utils.decorators import deprecated
from tfwrapper.utils.exceptions import raise_exception
from tfwrapper.utils.exceptions import InvalidArgumentException

from .dataset_generator import DatasetGenerator


def batch_data(data, batch_size):
    batches = []

    for i in range(0, int(len(data) / batch_size) + 1):
        start = (i * batch_size)
        end = min((i + 1) * batch_size, len(data))
        if start < end:
            batches.append(data[start:end])

    return batches


def normalize_array(arr):
    return (arr - arr.mean()) / arr.std()


def normalize_array_columnwise(arr):
    normalized = np.zeros(arr.shape)
    for col in range(arr.shape[1]):
        scaled =normalize_array(arr[:,col])
        normalized[:,col] = scaled

    return normalized


def shuffle_dataset(X, y, seed=None):
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(X))
    np.random.shuffle(idx)

    return X[idx], y[idx]


def balance_dataset(X, y, max_val=0):
    assert len(X) == len(y)

    is_onehot = False
    if len(y.shape) > 1 and y.shape[-1] > 1:
        is_onehot = True
        y = [np.argmax(y_) for y_ in y]

    counts = Counter(y)
    min_count = min([counts[x] for x in counts])
    if max_val is not 0:
        min_count = max_val

    counters = {}
    for val in y:
        counters[val] = 0

    balanced_X = []
    balanced_y = []

    for i in range(0, len(X)):
        if counters[y[i]] < min_count:
            try:
                balanced_X.append(X[i])
                balanced_y.append(y[i])
            except Exception as e:
                print(e)
        counters[y[i]] = counters[y[i]] + 1

    if is_onehot:
        y = onehot_array(y)

    return np.asarray(balanced_X), np.asarray(balanced_y)


def onehot_array(arr):
    shape = (len(arr), np.amax(arr) + 1)
    onehot = np.zeros(shape)
    for i in range(len(arr)):
        onehot[i][arr[i]] = 1

    return np.asarray(onehot)


def labels_to_indexes(y, sort=True):
    labels = []
    indices = []

    if sort:
        for label in y:
            if label not in labels:
                labels.append(label)
        labels = sorted(labels)

    for label in y:
        if label not in labels:
            labels.append(label)
        indices.append(labels.index(label))

    return np.asarray(indices), np.asarray(labels)


def split_dataset(X, y, ratio=0.8):
    train_len = int(len(X) * ratio)
    train_X = X[:train_len]
    train_y = y[:train_len]
    test_X = X[train_len:]
    test_y = y[train_len:]

    return np.asarray(train_X), np.asarray(train_y), np.asarray(test_X), np.asarray(test_y)


def parse_datastructure(root, allowed_suffixes=['jpg', 'jpeg', 'png'], verbose=False):
    X = []
    y = []

    i = 0

    for foldername in os.listdir(root):
        src = os.path.join(root, foldername)
        if os.path.isdir(src):
            for filename in os.listdir(src):
                if filename.lower().split('.')[-1] in allowed_suffixes:
                    src_file = os.path.join(src, filename)
                    X.append(src_file)
                    y.append(foldername)

                    if i % 1000 == 0:
                        logger.debug('Read %d images from %s' % (i, root))
                    i += 1
                elif verbose:
                    logger.warning('Skipping filename ' + filename)
        elif verbose:
            logger.warning('Skipping foldername ' + foldername)

    logger.info('Read %d images from %s' % (len(X), root))

    return np.asarray(X), np.asarray(y)


def parse_folder_with_labels_file(root, labels_file, verbose=False):
    X = []
    y = []

    i = 0
    with open(labels_file, 'r') as f:
        for line in f.readlines():
            label, filename = line.split(',')
            src_file = os.path.join(root, filename).strip()
            if os.path.isfile(src_file):
                X.append(src_file)
                y.append(label)
                
                if i % 1000 == 0:
                    logger.debug('Read %d images from %s' % (i, root))
                i += 1
            elif verbose:
                logger.warning('Skipping filename ' + src_file)

        logger.info('Read %d images from %s' % (len(X), root))

    return np.asarray(X), np.asarray(y)


def drop_classes(X, y, *, keep):
    filtered_X = []
    filtered_y = []

    for i in range(len(X)):
        if y[i] in keep:
            try:
                filtered_X.append(X[i])
                filtered_y.append(y[i])
            except Exception as e:
                print(e)

    return np.asarray(filtered_X), np.asarray(filtered_y)


class Dataset():

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def shape(self):
        return self._X.shape

    def __init__(self, X=np.asarray([]), y=np.asarray([]), paths=None, features=None, features_file=None, **kwargs):
        try:
            self._X = np.asarray(X)
            self._y = np.asarray(y)
        except Exception:
            self._X = X
            self._y = y
            logger.warning('Datatypes not compatible with numpy will be deprecated from Dataset soon(ish)')

        self.paths = paths
        
        if 'labels' in kwargs:
            self.labels = kwargs['labels']
            del kwargs['labels']
        else:
            self.labels = np.asarray([])

        if len(kwargs) > 0:
            errormsg = 'Invalid key(s) for dataset: %s' % [str(x) for x in kwargs]
            logger.error(errormsg)
            raise ValueError(errormsg)

        if features_file is not None:
            parsed_features = parse_features(features_file)
            self._X = np.asarray(parsed_features['features'].tolist())
            self._y = np.asarray(parsed_features['label'].tolist())

        if features is not None:
            self._X = np.asarray(features['features'].tolist())
            self._y = np.asarray(features['label'].tolist())

    def batch_generator(self, batch_size, normalize=False, shuffle=False, infinite=False):
        return DatasetGenerator(self, batch_size, normalize=normalize, shuffle=shuffle, infinite=infinite)
    
    def next_batch(self, start, batch_size):
        end = start + batch_size
        return self[start:end], end

    @deprecated('normalized')
    def normalize(self):
        return self.__class__(X=normalize_array(self._X), y=self._y, **self.kwargs())

    def normalized(self, columnwise=False):
        if columnwise:
            X = normalize_array_columnwise(self._X)
        else:
            X = normalize_array(self._X)

        return self.__class__(X=X, y=self._y, **self.kwargs())

    @deprecated('shuffled')
    def shuffle(self, seed=None):
        X, y = shuffle_dataset(self._X, self._y, seed=seed)

        return self.__class__(X=X, y=y, **self.kwargs())

    def shuffled(self, seed=None):
        X, y = shuffle_dataset(self._X, self._y, seed=seed)

        return self.__class__(X=X, y=y, **self.kwargs())

    @deprecated('balanced')
    def balance(self, max=0):
        X, y = balance_dataset(self._X, self._y, max_val=max)

        return self.__class__(X=X, y=y, **self.kwargs())

    def balanced(self, max=0):
        X, y = balance_dataset(self._X, self._y, max_val=max)

        return self.__class__(X=X, y=y, **self.kwargs())

    @deprecated('translated_labels')
    def translate_labels(self):
        y, labels = labels_to_indexes(self._y)

        return self.__class__(X=self._X, y=y, **self.kwargs(labels=labels))

    def translated_labels(self):
        y, labels = labels_to_indexes(self._y)

        return self.__class__(X=self._X, y=y, **self.kwargs(labels=labels))

    @deprecated('onehot_encoded')
    def onehot(self):
        return self.__class__(X=self._X, y=onehot_array(self._y), **self.kwargs())

    def onehot_encoded(self):
        y = self._y

        invalid_types = ['<U5', '<U11', '<U15', np.object, np.str, str]
        if y.dtype in invalid_types:
            y, self.labels = labels_to_indexes(y)

        try:
            return self.__class__(X=self._X, y=onehot_array(y), **self.kwargs())
        except TypeError:
            raise_exception('Invalid type for onehot_encoded %s. (Valid are all types of ints. Automatically converted are %s' % (y.dtype, str(invalid_types)), InvalidArgumentException)

    def split(self, ratio=0.8):
        X, y, test_X, test_y = split_dataset(self._X, self._y, ratio=ratio)
        train_dataset = self.__class__(X=X, y=y, **self.kwargs())
        test_dataset = self.__class__(X=test_X, y=test_y, **self.kwargs())

        return train_dataset, test_dataset

    def get_clusters(self, get_labels=False):
        if self.labels:
            raise NotImplementedError('Getting clusters with onehotted data not implemented')

        labels = sorted(list(set(self._y)))

        clusters = []
        for i in range(len(labels)):
            clusters.append(None)

        for i in range(len(self)):
            index = labels.index(self._y[i])
            if clusters[index] is None:
                clusters[index] = np.asarray([self._X[i]])
            else:
                clusters[index] = np.concatenate([clusters[index], np.asarray([self._X[i]])])

        if get_labels:
            return clusters, labels
        else:
            return clusters

    def drop_classes(self, *, drop=None, keep=None):
        if drop is None and keep is None:
            raise ValueError('When dropping classes, either a list of classes to drop or a list of classes to keep must be supplied')
        elif drop is None:
            _keep = set(self._y) & set(keep)
        else:
            _keep = set(self._y) - set(drop)

        X, y = drop_classes(self._X, self._y, keep=_keep)

        return self.__class__(X=X, y=y, **self.kwargs())

    def merge_classes(self, mappings):
        X = self._X
        y = self._y
        for i in range(len(y)):
            if y[i] in mappings:
                y[i] = mappings[y[i]]

        return self.__class__(X=X, y=y, **self.kwargs())

    def folds(self, k):
        X = np.array_split(self._X, k)
        y = np.array_split(self._y, k)

        folds = []
        for i in range(len(X)):
            folds.append(self.__class__(X=X[i], y=y[i], **self.kwargs()))

        return folds

    def kwargs(self, **kwargs):
        if not 'labels' in kwargs:
            kwargs['labels'] = self.labels

        return kwargs

    def __len__(self):
        return len(self._X)

    def __add__(self, other):
        X = np.concatenate((self._X, other._X))
        y = np.concatenate((self._y, other._y))
        labels = np.asarray([])
        if len(self.labels) > 0 and len(other.labels) > 0:
            labels = np.concatenate((self.labels, other.labels))

        return self.__class__(X=X, y=y, **self.kwargs(labels=labels))

    def _get_ndarray_slice(self, arr):
        return self._X[arr], self._y[arr]

    def __getitem__(self, val):
        valid_types = [int, slice, np.ndarray, list]
        if type(val) not in valid_types:
            raise_exception('Invalid slice datatype %s. (Valid is %s)' % (str(type(val)), str(valid_types)), NotImplementedError)

        if type(val) is np.ndarray:
            X, y = self._get_ndarray_slice(val)
        elif type(list) is list:
            X, y = self._get_ndarray_slice(np.asarray(list))
        else:
            X = self._X[val]
            y = self._y[val]

        labels = self.labels

        if type(val) is int:
            X = np.asarray([X])
            y = np.asarray([y])

        return self.__class__(X=X, y=y, **self.kwargs(labels=labels))
