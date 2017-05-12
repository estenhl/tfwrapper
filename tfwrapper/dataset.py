import os
import cv2
import scipy.ndimage
import numpy as np
import tensorflow as tf
import pandas as pd
from random import shuffle
from collections import Counter

from .logger import logger
from .tfsession import TFSession
from tfwrapper import twimage
from tfwrapper.utils.data import parse_features
from tfwrapper.utils.data import write_features

def normalize_array(arr):
    return (arr - arr.mean()) / arr.std()

def shuffle_dataset(X, y):
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

def parse_datastructure(root, suffix='.jpg', verbose=False):
    X = []
    y = []

    for foldername in os.listdir(root):
        src = os.path.join(root, foldername)
        if os.path.isdir(src):
            for filename in os.listdir(src):
                if filename.lower().endswith(suffix):
                    src_file = os.path.join(src, filename)
                    X.append(src_file)
                    y.append(foldername)
                elif verbose:
                    print('Skipping filename ' + filename)
        elif verbose:
            print('Skipping foldername ' + foldername)

    return np.asarray(X), np.asarray(y)

def parse_folder_with_labels_file(root, labels_file, verbose=False):
    X = []
    y = []

    with open(labels_file, 'r') as f:
        for line in f.readlines():
            label, filename = line.split(',')
            src_file = os.path.join(root, filename).strip()
            if os.path.isfile(src_file):
                X.append(src_file)
                y.append(label)
            elif verbose:
                print('Skipping filename ' + src_file)

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

class DatasetGenerator():
    def __init__(self, dataset, batch_size, normalize=False, shuffle=False, infinite=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.normalize = normalize
        self.shuffle = shuffle
        self.infinite = infinite
        self.cursor = 0

    def __iter__(self):
        self.cursor = 0
        return self

    def __next__(self):
        if self.cursor >= len(self):
            if not self.infinite:
                raise StopIteration
            
            self.cursor = 0

        dataset, self.cursor = self.dataset.next_batch(self.cursor, self.batch_size)

        if self.normalize:
            dataset = dataset.normalize()

        if self.shuffle:
            dataset = dataset.shuffle()

        return dataset.X, dataset.y

    def __len__(self):
        return len(self.dataset)

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

    def __init__(self, X=np.asarray([]), y=np.asarray([]), features=None, features_file=None, verbose=False, **kwargs):
        self._X = X
        self._y = y
        
        if 'labels' in kwargs:
            self.labels = kwargs['labels']
        else:
            self.labels = np.asarray([])

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

    def normalize(self):
        return self.__class__(X=normalize_array(self._X), y=self._y, **self.kwargs())

    def shuffle(self):
        X, y = shuffle_dataset(self._X, self._y)

        return self.__class__(X=X, y=y, **self.kwargs())

    def balance(self, max=0):
        X, y = balance_dataset(self._X, self._y, max_val=max)

        return self.__class__(X=X, y=y, **self.kwargs())

    def translate_labels(self):
        y, labels = labels_to_indexes(self._y)

        return self.__class__(X=self._X, y=y, **self.kwargs(labels=labels))

    def onehot(self):
        return self.__class__(X=self._X, y=onehot_array(self._y), **self.kwargs())

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
            raise InvalidArgumentException('When dropping classes, either a list of classes to drop or a list of classes to keep must be supplied')
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

    def __getitem__(self, val):
        if type(val) not in [slice, int]:
            raise NotImplementedError('Dataset only handles slices and ints')

        X = self._X[val]
        y = self._y[val]
        labels = np.asarray([])
        if len(self.labels) > 0:
            labels = self.labels[val]

        if type(val) is int:
            X = np.asarray([X])
            y = np.asarray([y])

        return self.__class__(X=X, y=y, **self.kwargs(labels=labels))

class ImageDataset(Dataset):
    loaded_X = None
    loaded_y = None

    @property
    def X(self):
        if self.loaded_X is not None:
            return self.loaded_X

        dataset, _ = self.next_batch(0, float('inf'))
        
        self.loaded_X = dataset.X
        self.loaded_y = dataset.y
        
        return dataset.X

    @property
    def y(self):
        if self.loaded_y is not None:
            return self.loaded_y

        dataset, _ = self.next_batch(0, float('inf'))
        
        self.loaded_X = dataset.X
        self.loaded_y = dataset.y
        
        return dataset.y

    @property
    def shape(self):
        if self.loaded_X is not None:
            return self.loaded_X.shape
        else:
            return [len(self)] + self.loader.shape[1:]

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, value):
        self.loaded_X = None
        self.loaded_y = None
        self._loader = value

    def __init__(self, X=np.asarray([]), y=np.asarray([]), root_folder=None, labels_file=None, **kwargs):
        _X = X
        _y = y

        if 'loader' in kwargs:
            self._loader = kwargs['loader']
        else:
            self._loader = ImageLoader()
            
        if labels_file is not None and root_folder is not None:
            _X, _y = parse_folder_with_labels_file(root_folder, labels_file)
        elif root_folder is not None:
            _X, _y = parse_datastructure(root_folder)

        super().__init__(X=_X, y=_y, **kwargs)

    def normalize(self):
        raise NotImplementedError('Unable to normalize an imagedataset which is read on-demand')

    def next_batch(self, cursor, batch_size):
        batch_X = []
        batch_y = []

        while len(batch_X) < batch_size and cursor < len(self):
            imgs, names = self.loader.load(self._X[cursor], label=self._y[cursor])
            batch_X += imgs
            batch_y += [self._y[cursor]] * len(imgs)
            cursor += 1

        X = np.asarray(batch_X)
        y = np.asarray(batch_y)

        return Dataset(X=X, y=y), cursor

    def kwargs(self, **kwargs):
        kwargs = super().kwargs(**kwargs)
        kwargs['loader'] = self.loader

        return kwargs

ROTATED = 'rotated'
ROTATION_STEPS = 'rotation_steps'
MAX_ROTATION_ANGLE = 'max_rotation_angle'
BLURRED = 'blurred'
BLUR_STEPS = 'blur_steps'
MAX_BLUR_SIGMA = 'max_blur_sigma'

def create_name(name, suffixes):
    return "_".join([name] + suffixes)

class ImagePreprocessor():
    resize_to = False
    bw = False
    flip_lr = False
    flip_ud = False
    blur = False
    rotate = False

    augs = {}

    def rotate(self, rotation_steps=1, max_rotation_angle=10):
        self.augs[ROTATED] = {ROTATION_STEPS: rotation_steps, MAX_ROTATION_ANGLE: max_rotation_angle}
        self.augs[ROTATION_STEPS] = rotation_steps
        self.augs[MAX_ROTATION_ANGLE] = max_rotation_angle

    def blur(self, blur_steps=1, max_blur_sigma=1):
        self.augs[BLURRED] = {BLUR_STEPS: blur_steps, MAX_BLUR_SIGMA: max_blur_sigma}

    def process(self, img, name, label=None):
        if img is None:
            return [], []
        
        imgs = []
        names = []

        org_suffixes = []

        if self.resize_to:
            img = twimage.resize(img, self.resize_to)
            #Should check for size
            width, height = self.resize_to
            org_suffixes.append('%s%dx%d' % ('resize', width, height))
        if self.bw:
            img = twimage.bw(img, shape=3)
            org_suffixes.append('bw')

        imgs.append(img)
        names.append(create_name(name, org_suffixes))

        if self.flip_lr:
            imgs.append(np.fliplr(img))
            org_suffixes.append('fliplr')
            names.append(create_name(name, org_suffixes))
            org_suffixes.remove('fliplr')

        if self.flip_ud:
            imgs.append(np.flipud(img))
            org_suffixes.append('flipud')
            names.append(create_name(name, org_suffixes))
            org_suffixes.remove('flipud')

        if ROTATED in self.augs:
            rotation_steps = self.augs[ROTATED][ROTATION_STEPS]
            max_rotation_angle = self.augs[ROTATED][MAX_ROTATION_ANGLE]
            for i in range(rotation_steps):
                angle = max_rotation_angle * (i+1)/rotation_steps
                imgs.append(twimage.rotate(img, angle))
                org_suffixes.append(ROTATED)
                org_suffixes.append(str(angle))
                names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(angle))

                imgs.append(twimage.rotate(img, -angle))
                org_suffixes.append(str(-angle))
                names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(-angle))
                org_suffixes.remove(ROTATED)

        if BLURRED in self.augs:
            blur_steps = self.augs[BLURRED][BLUR_STEPS]
            max_blur_sigma = self.augs[BLURRED][MAX_BLUR_SIGMA]
            for i in range(blur_steps):
                sigma = max_blur_sigma * (i+1)/blur_steps
                imgs.append(twimage.blur(img, sigma))
                org_suffixes.append(BLURRED)
                org_suffixes.append(str(sigma))
                names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(sigma))
                org_suffixes.remove(BLURRED)

        # TODO: generate combinations of flip, rotation and blur

        return imgs, names

class ImageLoader():

    @property
    def shape(self):
        if self.preprocessor.resize_to:
            return [-1] + list(self.preprocessor.resize_to) + [3]

        return [-1, -1, -1, 3]

    def __init__(self, preprocessor=ImagePreprocessor()):
        self.preprocessor = preprocessor

    def load(self, path, name=None, label=None):
        if name is None:
            name = '.'.join(os.path.basename(path).split('.')[:-1])

        img = twimage.imread(path)
        return self.preprocessor.process(img, name, label=label)

class FeatureLoader(ImageLoader):
    sess = None

    @property
    def shape(self):
        layer = self.layer

        if layer is None:
            layer = -1

        return self.model.get_layer_shape(layer)

    def __init__(self, model, layer=None, cache=None, preprocessor=ImagePreprocessor(), sess=None):
        super().__init__(preprocessor=preprocessor)
        self.model = model
        self.layer = layer

        self.cache = None
        self.features = pd.DataFrame(columns = ['filename', 'label', 'features'])
        if cache:
            self.cache = cache
            self.features = parse_features(cache)

        self.sess = sess

    def load(self, img, name=None, label=None):
        if label is not None and type(label) is np.ndarray:
            label = np.argmax(label)

        imgs, names = super().load(img, name, label=label)
        features = []
        records = []

        with TFSession(self.sess) as sess:
            for i in range(len(imgs)):
                if names[i] in self.features['filename'].values:
                    logger.info('Skipping %s' % names[i])
                    vector = self.features[self.features['filename'] == names[i]]['features']
                    vector = np.asarray(vector)[0]
                    features.append(vector)
                else:
                    logger.info('Extracting features for %s' % names[i])
                    if self.layer is None:
                        vector = self.model.extract_bottleneck_features(imgs[i], sess=sess)
                    else:
                        vector = self.model.extract_features(imgs[i], layer=self.layer, sess=sess)
                    features.append(vector)
                    record = {'filename': names[i], 'features': vector}

                    if label is not None:
                        record['label'] = label

                    records.append(record)
                    self.features = self.features.append(record, ignore_index=True)

            if self.cache and len(records) > 0:
                write_features(self.cache, records, append=os.path.isfile(self.cache))

        return features, names
