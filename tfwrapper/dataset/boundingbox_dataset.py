import math
import os
import numpy as np
import cv2

from tfwrapper import twimage

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

def _parse_fddb_annotations(root_folder, labels_folder, size=None):
    X = []
    y = []

    parsed = 0
    for labels_file in os.listdir(labels_folder):
        if labels_file != '.DS_Store':
            completed = False
            with open(os.path.join(labels_folder, labels_file), 'r') as f:
                lines = f.readlines()
                i = 0

                while i < len(lines):
                    filename = lines[i].strip() + '.jpg'
                    img = twimage.imread(os.path.join(root_folder, filename))
                    parsed += 1
                    i += 1

                    num_boxes = int(lines[i])
                    i += 1

                    boxes = []
                    for j in range(num_boxes):
                        tokens = [float(x) for x in lines[i + j].strip().split()]
                        y_min, x_min, y_max, x_max = _parse_fddb_region(tokens)
                        

                        boxes.append(['face', [y_min, x_min, y_max, x_max]])

                    y.append(boxes)
                    X.append(img)
                    i += num_boxes

                    if size is not None and parsed == size:
                        completed = True
                        break

        if completed:
            break

    return np.asarray(X), np.asarray(y)


class BoundingBoxDataset(Dataset):
    def __init__(self, X, y):
        super().__init__(X=X, y=y)

    @classmethod
    def from_fddb_annotations(cls, *, root_folder, labels_folder, size=None):
        X, y = _parse_fddb_annotations(root_folder, labels_folder, size=size)

        return cls(X=X, y=y)

    def visualize(self, num=None):
        if num is None:
            num = len(self)
        elif num > len(self):
            logger.warning('Unable to visualize a larger number of items then the dataset contains')
            num = len(self)

        for i in range(num):
            img = self._X[i].copy()
            for _, (y_min, x_min, y_max, x_max) in self._y[i]:
                img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=3)
            twimage.show(img)