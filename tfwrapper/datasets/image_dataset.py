import os
from random import shuffle
import numpy as np
class ImageDataset():
    def __init__(self, dir_path=None, names=None, labels=None, img_path=None):
        if dir_path is None:
            self.names = names
            self.labels = labels
            self.img_path = img_path
        else:
            self.img_path, self.labels, self.names = self.read_features_from_dir(dir_path)

    def read_features_from_dir(self, root, suffix='.jpg', verbose=False):
        img_path = []
        y = []
        names = []

        for foldername in os.listdir(root):
            src = os.path.join(root, foldername)
            if os.path.isdir(src):
                for filename in os.listdir(src):
                    if filename.endswith(suffix):
                        src_file = os.path.join(src, filename)
                        img_path.append(src_file)
                        y.append(foldername)
                        names.append(filename)
                    elif verbose:
                        print('Skipping filename ' + filename)
            elif verbose:
                print('Skipping foldername ' + foldername)

        return img_path, y, names

    def length(self):
        return len(self.names)

    def shuffle(self):
        index_shuf = list(range(len(self.names)))
        shuffle(index_shuf)
        self.names = [self.names[i] for i in index_shuf]
        self.labels = [self.labels[i] for i in index_shuf]
        self.img_path = [self.img_path[i] for i in index_shuf]

    def split(self, shape=[0.9, 0.1]):
        size = len(self.names)

        split_index = int(size * shape[0])
        dataset1 = ImageDataset(
            names=self.names[:split_index],
            labels=self.labels[:split_index],
            img_path=self.img_path[:split_index]
        )
        dataset2 = ImageDataset(
            names=self.names[split_index:],
            labels=self.labels[split_index:],
            img_path=self.img_path[split_index:]
        )
        if len(shape) == 3:
            normalize = shape[0] * 0.5
            dataset2, dataset3 = dataset2.split(shape=[shape[1] + normalize, shape[2] + normalize])
            return dataset1, dataset2, dataset3

        return dataset1, dataset2

    def one_hot_encode(self):
        self.one_hot = OneHot(self.labels)

    def get_data(self, one_hot=True):
        if self.one_hot:
            labels = self.one_hot.one_hot_encode(self.labels)
        else:
            labels = self.labels
        return self.names, self.img_path, labels



class OneHot():
    def __init__(self, labels):
        unique = list(set(labels))
        self.labels = sorted(unique)
        self.size = len(self.labels)

        self.label_to_onehot = {}
        self.onehot_to_label = {}

        for i, label in enumerate(self.labels):
            self.label_to_onehot[label] = i
            self.onehot_to_label[i] = label


    def one_hot_encode(self, labels):
        number_value = [self.label_to_onehot[label] for label in labels]

        one_hots = np.zeros([len(labels), self.size])
        for i, number in enumerate(number_value):
            one_hots[i][number] = 1
        return one_hots.tolist()