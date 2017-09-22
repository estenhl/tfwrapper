import cv2
import hashlib
import io
import json
import math
import os
import numpy as np
import tensorflow as tf

from tfwrapper import twimage
from tfwrapper.utils.data import create_tfrecord_feature
from tfwrapper.utils.exceptions import log_and_raise

from .dataset import Dataset


def _parse_fddb_region(region):
    major_axis_radius = region[0]
    minor_axis_radius = region[1]
    angle = region[2] + (math.pi / 2)
    x_center = region[3]
    y_center = region[4]

    y_diff = math.sqrt((major_axis_radius ** 2) * (math.cos(angle) ** 2) + (minor_axis_radius ** 2) * (math.sin(angle) ** 2))
    y_max = y_center + y_diff 
    y_min = y_center - y_diff

    x_diff = math.sqrt((major_axis_radius ** 2) * (math.sin(angle) ** 2) + (minor_axis_radius ** 2) * (math.cos(angle) ** 2))
    x_max = x_center + x_diff
    x_min = x_center - x_diff

    return int(y_min), int(x_min), int(y_max), int(x_max)


def _parse_fddb_annotations(root_folder, labels_folder, size=None, label='face'):
    X = []
    y = []
    paths = []

    parsed = 0
    for labels_file in os.listdir(labels_folder):
        if labels_file != '.DS_Store':
            completed = False
            with open(os.path.join(labels_folder, labels_file), 'r') as f:
                lines = f.readlines()
                i = 0

                while i < len(lines):
                    filename = lines[i].strip() + '.jpg'
                    path = os.path.join(root_folder, filename)
                    img = twimage.imread(path)
                    parsed += 1
                    i += 1

                    num_boxes = int(lines[i])
                    i += 1

                    boxes = []
                    for j in range(num_boxes):
                        tokens = [float(x) for x in lines[i + j].strip().split()]
                        y_min, x_min, y_max, x_max = _parse_fddb_region(tokens)
                        

                        boxes.append([label, [y_min, x_min, y_max, x_max]])

                    X.append(img)
                    y.append(boxes)
                    paths.append(path)
                    i += num_boxes

                    if size is not None and parsed == size:
                        completed = True
                        break

        if completed:
            break

    return np.asarray(X), np.asarray(y), np.asarray(paths)


def _parse_vgg_json(root_folder, annotations_file):
    X = []
    y = []
    paths = []

    with open(annotations_file, 'r') as f:
        data = json.load(f)

    for key in data:
        entry = data[key]
        filename = entry['filename']
        path = os.path.join(root_folder, filename)
        img = twimage.imread(path)

        boxes = []
        for i in entry['regions']:
            region = entry['regions'][i]
            label = region['region_attributes']['label']

            shape_attrs = region['shape_attributes']
            if shape_attrs['name'] != 'rect':
                logger.warning('Unable to parse vgg region with type %s. (Only \'rect\' is supported). Skipping region %s for %s' % (shape_attrs['name'], str(i), filename))
                continue

            y_min = int(shape_attrs['y'])
            x_min = int(shape_attrs['x'])
            y_max = y_min + int(shape_attrs['height'])
            x_max = x_min + int(shape_attrs['width'])

            boxes.append([label, [y_min, x_min, y_max, x_max]])

        X.append(img)
        y.append(boxes)
        paths.append(path)

    return np.asarray(X), np.asarray(y), np.asarray(paths)


def _translate_boundingbox_labels(y):
    labels = []
    
    for entry in y:
        for label, _ in entry:
            if label not in labels:
                labels.append(label)

    labels = sorted(labels)
    labels = ['background'] + labels

    translated_y = []
    for entry in y:
        new_entry = []
        for label, boundingbox in entry:
            new_entry.append([labels.index(label), boundingbox])
        translated_y.append(new_entry)

    return np.asarray(translated_y), np.asarray(labels)


def _shuffle_boundingbox_dataset(X, y, paths, seed=None):
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(X))
    np.random.shuffle(idx)

    return X[idx], y[idx], paths[idx]


class BoundingBoxDataset(Dataset):
    """ Stores datasets where the labels are bounding boxes.

    Attributes
    ----------
    X : np.ndarray
        A 4-dimensional array containing images

    y : np.ndarray
        An ndarray of lists where each entry is a list of bounding boxes related to an image. The list is on the form:
        [
            [label, [ymin, xmin, ymax, xmax]],
            [label, [ymin, xmin, ymax, xmax]]
        ]

    labels : np.ndarray
        A 1-dimensional array containing labels related to the label tags of the bounding boxes

    paths : np.ndarray
        A 1-dimensional ndarray containing the paths of the images in X

    """


    def __init__(self, X, y, paths=None, **kwargs):
        super().__init__(X=X, y=y, **kwargs)
        
        if paths is None:
            paths = np.asarray([])

        self.paths = paths

    @classmethod
    def from_fddb_annotations(cls, *, root_folder, labels_folder, size=None):
        X, y, paths = _parse_fddb_annotations(root_folder, labels_folder, size=size)

        return cls(X=X, y=y, paths=paths)

    @classmethod
    def from_vgg_json(cls, *, root_folder, annotations_file):
        X, y, paths = _parse_vgg_json(root_folder, annotations_file)

        return cls(X=X, y=y, paths=paths)

    def translated_labels(self):
        y, labels = _translate_boundingbox_labels(self._y)

        return self.__class__(self._X, y, paths=self.paths, labels=labels)

    # TODO: THESE THREE (shuffle and shuffled, split) SHOULD USE SOME GENERIC SHUFFLING FUNCTION IN THE SUPERCLASS
    def shuffle(self, seed=None):
        log_and_raise(NotImplementedException, 'Use BoundingBoxDataset.shuffled')

    def shuffled(self, seed=None):
        if self.paths is None or len(self.paths) == 0:
            return super().shuffled(seed=seed)

        X, y, paths = _shuffle_boundingbox_dataset(self._X, self._y, self.paths, seed=seed)

        return self.__class__(X=X, y=y, paths=paths)

    def split(self, ratio):
        pivot = int(len(self) * ratio)

        return self.__class__(X=self._X[:pivot], y=self._y[:pivot], paths=self.paths[:pivot], labels=self.labels), self.__class__(X=self._X[pivot:], y=self.y[pivot:], paths=self.paths[pivot:], labels=self.labels)

    def kwargs(self, **kwargs):
        kwargs = super().kwargs(**kwargs)
        if 'paths' not in kwargs:
            kwargs['paths'] = self.paths

        return kwargs

    def visualize(self, num=None):
        if num is None:
            num = len(self)
        elif num > len(self):
            logger.warning('Unable to visualize a larger number of items then the dataset contains')
            num = len(self)

        for i in range(num):
            img = self._X[i].copy()
            for label, (y_min, x_min, y_max, x_max) in self._y[i]:
                img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=3)
                if not hasattr(self, 'labels') or self.labels is None or len(self.labels) == 0:
                    img = cv2.putText(img, str(label), (x_min, y_min-10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color=(0, 255, 0))
                else:
                    img = cv2.putText(img, '%s: %s' % (str(label), self.labels[label]), (x_min, y_min-10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color=(0, 255, 0))
            twimage.show(img)

    def to_tfrecord(self, output_path):
        with tf.python_io.TFRecordWriter(output_path) as writer:
            for i in range(len(self._X)):
                path = self.paths[i]
                img = self._X[i]
                height, width, _ = img.shape

                bboxes = self._y[i]

                # TODO (21.09.17): This seems very unecessary (should find out if we can use some sort of numpy to bytestring)
                with tf.gfile.GFile(path, 'rb') as fid:
                    encoded_jpg = fid.read()
                encoded_img = io.BytesIO(encoded_jpg)
                key = hashlib.sha256(encoded_jpg).hexdigest()

                ymins = []
                xmins = []
                ymaxs = []
                xmaxs = []
                label_names = []
                labels = []

                for label, (ymin, xmin, ymax, xmax) in bboxes:
                    ymins.append(max(0., float(ymin / height)))
                    xmins.append(max(0., float(xmin / width)))
                    ymaxs.append(min(1., float(ymax / height)))
                    xmaxs.append(min(1., float(xmax / width)))
                    label_names.append(self.labels[label].encode('utf8'))
                    labels.append(label)

                featuremap = {
                    'image/height': create_tfrecord_feature(height),
                    'image/width': create_tfrecord_feature(width),
                    'image/filename': create_tfrecord_feature(str.encode(path)),
                    'image/source_id': create_tfrecord_feature(str.encode(path)),
                    'image/key/sha256': create_tfrecord_feature(str.encode(key)),
                    'image/encoded': create_tfrecord_feature(encoded_jpg),
                    'image/format': create_tfrecord_feature(str.encode('jpeg')),
                    'image/object/bbox/ymin': create_tfrecord_feature(ymins),
                    'image/object/bbox/xmin': create_tfrecord_feature(xmins),
                    'image/object/bbox/ymax': create_tfrecord_feature(ymaxs),
                    'image/object/bbox/xmax': create_tfrecord_feature(xmaxs),
                    'image/object/class/text': create_tfrecord_feature(label_names),
                    'image/object/class/label': create_tfrecord_feature(labels),

                    # TODO (14.09.17): These should be handled better
                    'image/object/difficult': create_tfrecord_feature([]),
                    'image/object/truncated': create_tfrecord_feature([]),
                    'image/object/view': create_tfrecord_feature([])
                }

                data = tf.train.Example(features=tf.train.Features(feature=featuremap))

                writer.write(data.SerializeToString())


    def __add__(self, other, **kwargs):
        kwargs['paths'] = np.concatenate([self.paths, other.paths])

        return super().__add__(other, **kwargs)
