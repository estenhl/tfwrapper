import os
import cv2
import scipy.ndimage
import numpy as np
from collections import Counter
from random import shuffle

from tfwrapper.utils.data import parse_features
from tfwrapper.utils.data import write_features

def normalize_array(arr):
    return (arr - arr.mean()) / arr.std()

def shuffle_dataset(X, y):
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    return np.squeeze(X[idx]), np.squeeze(y[idx])

def balance_dataset(X, y, max=float('inf')):
    assert len(X) == len(y)

    is_onehot = False
    if len(y.shape) > 1 and y.shape[-1] > 1:
        is_onehot = True
        y = [np.argmax(y_) for y_ in y]

    counts = Counter(y)
    min_count = min([counts[x] for x in counts])
    min_count = min(min_count, max)

    counters = {}
    for val in y:
        counters[val] = 0

    balanced_X = []
    balanced_y = []

    for i in range(0, len(X)):
        if counters[y[i]] < min_count:
            balanced_X.append(X[i])
            balanced_y.append(y[i])
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
                if filename.endswith(suffix):
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
            filtered_X.append(X[i])
            filtered_y.append(y[i])

    return np.asarray(filtered_X), np.asarray(filtered_y)

class Dataset():

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    def __init__(self, parent=None, X=np.asarray([]), y=np.asarray([]), labels=np.asarray([]), features=None, features_file=None, verbose=False):
        self._X = X
        self._y = y
        self.labels = labels

        if features_file is not None:
            parsed_features = parse_features(features_file)
            self._X = np.asarray(parsed_features['features'].tolist())
            self._y = np.asarray(parsed_features['label'].tolist())

        if features is not None:
            self._X = np.asarray(features['features'].tolist())
            self._y = np.asarray(features['label'].tolist())

    def normalize(self):
        return self.__class__(X=normalize_array(self._X), y=self._y, labels=self.labels)

    def shuffle(self):
        X, y = shuffle_dataset(self._X, self._y)

        return self.__class__(X=X, y=y, labels=self.labels)

    def balance(self, max=float('inf')):
        X, y = balance_dataset(self._X, self._y, max=max)

        return self.__class__(X=X, y=y, labels=self.labels)

    def translate_labels(self):
        y, labels = labels_to_indexes(self._y)

        return self.__class__(X=self._X, y=y, labels=labels)

    def onehot(self):
        return self.__class__(X=self._X, y=onehot_array(self._y), labels=self.labels)

    def split(self, ratio=0.8):
        X, y, test_X, test_y = split_dataset(self._X, self._y, ratio=ratio)
        train_dataset = self.__class__(X=X, y=y, labels=self.labels)
        test_dataset = self.__class__(X=test_X, y=test_y, labels=self.labels)

        return train_dataset, test_dataset

    def drop_classes(self, *, drop=None, keep=None):
        if drop is None and keep is None:
            raise InvalidArgumentException('When dropping classes, either a list of classes to drop or a list of classes to keep must be supplied')
        elif drop is None:
            _keep = set(self._y) & set(keep)
        else:
            _keep = set(self._y) - set(drop)

        X, y = drop_classes(self._X, self._y, keep=_keep)

        return self.__class__(X=X, y=y, labels=self.labels)

    def merge_classes(self, mappings):
        X = self._X
        y = self._y
        for i in range(len(y)):
            if y[i] in mappings:
                y[i] = mappings[y[i]]

        return self.__class__(X=X, y=y, labels=self.labels)

    def folds(self, k):
        X = np.array_split(self._X, k)
        y = np.array_split(self._y, k)

        folds = []
        for i in range(len(X)):
            folds.append(self.__class__(X=X[i], y=y[i], labels=self.labels))

        return folds

    def __len__(self):
        return len(self._X)

    def __add__(self, other):
        X = np.concatenate((self._X, other._X))
        y = np.concatenate((self._y, other._y))

        return self.__class__(X=X, y=y, labels=self.labels)

    def __getitem__(self, val):
        if type(val) not in [slice, int]:
            raise NotImplementedError('Dataset only handles slices and ints')

        X = self._X[val]
        y = self._y[val]

        if type(val) is int:
            X = np.asarray([X])
            y = np.asarray([y])

        return self.__class__(X=X, y=y)

from tfwrapper import twimage

RESIZE = "resize"
TRANSFORM_BW = "bw"
FLIP_LR = "fliplr"
FLIP_UD = "flipud"
ROTATED = 'rotated'
ROTATION_STEPS = 'rotation_steps'
MAX_ROTATION_ANGLE = 'max_rotation_angle'
BLURRED = 'blurred'
BLUR_STEPS = 'blur_steps'
MAX_BLUR_SIGMA = 'max_blur_sigma'

def create_name(name, suffixes):
    return "_".join([name] + suffixes)


class ImagePreprocess():
    resize_to = False
    bw = False
    flip_lr = False
    flip_ud = False

    # TODO: REMOVE
    augs = {}

    def rotate(self, rotation_steps=1, max_rotation_angle=10):
        self.augs[ROTATED] = {ROTATION_STEPS: rotation_steps, MAX_ROTATION_ANGLE: max_rotation_angle}
        self.augs[ROTATION_STEPS] = rotation_steps
        self.augs[MAX_ROTATION_ANGLE] = max_rotation_angle

    def blur(self, blur_steps=1, max_blur_sigma=1):
        self.augs[BLURRED] = {BLUR_STEPS: blur_steps, MAX_BLUR_SIGMA: max_blur_sigma}

    def apply_file(self, image_path, name=None, label=None):
        if name is None:
            name = '.'.join(os.path.basename(image_path).split('.')[:-1])

        img = twimage.imread(image_path)
        return self.apply(img, name, label=label)

    def apply(self, img, name, label=None):
        if img is None:
            return [], []
        
        img_versions = []
        img_names = []

        org_suffixes = []

        if self.resize_to:
            img = twimage.resize(img, self.resize_to)
            #Should check for size
            width, height = self.resize_to
            org_suffixes.append('%s%dx%d' % (RESIZE, width, height))
        if self.bw:
            img = twimage.bw(img, shape=3)
            org_suffixes.append(TRANSFORM_BW)

        img_versions.append(img)
        img_names.append(create_name(name, org_suffixes))

        # img_versions.append(img)
        # img_names.append(org_suffixes)
        # #Append
        if self.flip_lr:
            img_versions.append(np.fliplr(img))
            org_suffixes.append(FLIP_LR)
            img_names.append(create_name(name, org_suffixes))
            org_suffixes.remove(FLIP_LR)

        if self.flip_ud:
            img_versions.append(np.flipud(img))
            org_suffixes.append(FLIP_UD)
            img_names.append(create_name(name, org_suffixes))
            org_suffixes.remove(FLIP_UD)

        if ROTATED in self.augs:
            rotation_steps = self.augs[ROTATED][ROTATION_STEPS]
            max_rotation_angle = self.augs[ROTATED][MAX_ROTATION_ANGLE]
            for i in range(rotation_steps):
                angle = max_rotation_angle * (i+1)/rotation_steps
                img_versions.append(twimage.rotate(img, angle))
                org_suffixes.append(ROTATED)
                org_suffixes.append(str(angle))
                img_names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(angle))

                img_versions.append(twimage.rotate(img, -angle))
                org_suffixes.append(str(-angle))
                img_names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(-angle))
                org_suffixes.remove(ROTATED)

        if BLURRED in self.augs:
            blur_steps = self.augs[BLURRED][BLUR_STEPS]
            max_blur_sigma = self.augs[BLURRED][MAX_BLUR_SIGMA]
            for i in range(blur_steps):
                sigma = max_blur_sigma * (i+1)/blur_steps
                img_versions.append(twimage.blur(img, sigma))
                org_suffixes.append(BLURRED)
                org_suffixes.append(str(sigma))
                img_names.append(create_name(name, org_suffixes))
                org_suffixes.remove(str(sigma))
                org_suffixes.remove(BLURRED)

        # TODO: generate combinations of flip, rotation and blur

        return img_names, img_versions

import tensorflow as tf 

class FeatureExtractor(ImagePreprocess):
    def __init__(self, model, feature_file=None, sess=None):
        super().__init__()
        self.model = model
        self.feature_file = feature_file
        self.features = parse_features(feature_file)
        self.sess = sess

    def apply(self, img, name, label=None):
        if label is not None and type(label) is np.ndarray:
            label = np.argmax(label)

        names, imgs = super().apply(img, name, label=label)
        features = []
        records = []

        for i in range(len(imgs)):
            if names[i] in self.features['filename'].values:
                print('Skipping %s' % names[i])
                vector = self.features[self.features['filename'] == names[i]]['features']
                vector = np.asarray(vector)[0]
                features.append(vector)
            else:
                print('Extracting features for %s' % names[i])
                vector = self.model.get_feature(imgs[i], sess=self.sess)
                features.append(vector)
                record = {'filename': names[i], 'features': vector}

                if label is not None:
                    record['label'] = label

                records.append(record)
                self.features = self.features.append(record, ignore_index=True)

        if self.feature_file and len(records) > 0:
            write_features(self.feature_file, records, append=os.path.isfile(self.feature_file))

        return names, features

class ImageDataset(Dataset):
    preprocessor = ImagePreprocess()
    loaded_X = None
    loaded_y = None

    @property
    def X(self):
        if self.loaded_X is not None:
            return self.loaded_X

        X, y, _ = self.read_batch(0)
        self.loaded_X = X
        self.loaded_y = y
        return X

    @property
    def y(self):
        if self.loaded_y is not None:
            return self.loaded_y

        X, y, _ = self.read_batch(0)
        self.loaded_X = X
        self.loaded_y = y
        return y

    def __init__(self, X=np.asarray([]), y=np.asarray([]), labels=np.asarray([]), root_folder=None, labels_file=None):
        _X = X
        _y = y

        if labels_file is not None and root_folder is not None:
            _X, _y = parse_folder_with_labels_file(root_folder, labels_file)
        elif root_folder is not None:
            _X, _y = parse_datastructure(root_folder)

        self.cursor = 0
        super().__init__(X=_X, y=_y, labels=labels)

    def next_batch(self, batch_size=128):
        while self.cursor + batch_size < len(self):
            batch_X, batch_y, cursor = self.read_batch(self.cursor)
            self.cursor = cursor
            yield Dataset(X=batch_X, y=batch_y, labels=self.labels)

    def read_batch(self, cursor, batch_size=float('inf')):
        batch_X = []
        batch_y = []

        while len(batch_X) < batch_size and cursor < len(self):
            _, imgs = self.preprocessor.apply_file(self._X[cursor], label=self._y[cursor])
            batch_X += imgs
            batch_y += [self._y[cursor]] * len(imgs)
            cursor += 1

        X = np.asarray(batch_X)
        y = np.asarray(batch_y)

        return X, y, cursor
