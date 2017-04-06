import os
import numpy as np
import tensorflow as tf
from tfwrapper.datasets.image_augment import ImageAugment, ImagePreprocess
from tfwrapper.datasets.image_dataset import ImageDataset
from tfwrapper.nets.pretrained.pretrained_model import PretrainedModel
from tfwrapper.datasets import features

class BatchGenerator():

    def get_batch(self, batch_size):
        pass


class CachedFeatureGenerator(BatchGenerator):

    def __init__(self, file_path: str, model: PretrainedModel, layer=None):
        self.file_path = file_path
        self.model = model
        self.layer = layer

        self.cache = self.read_cache_from_file()

    def read_cache_from_file(self):
        return features.parse_features(self.file_path)

    def save_cache_to_file(self):
        features.write_features(self.cache, self.file_path)

    def load(self, dataset: ImageDataset, image_aug: ImagePreprocess):
        names, imgs, labels = image_aug.apply_dataset(dataset)

        features = []
        sess = tf.Session(graph=self.model.graph)

        counter_log_interval = len(names)/10
        print(len(names))
        print(counter_log_interval)
        for i, (name, img) in enumerate(zip(names, imgs)):
            print(name)
            if (i % counter_log_interval) == 0:
                print("{}% parsed".format(i / counter_log_interval))
            if name in self.cache:
                features.append(self.cache[name])
                print("Already cached")
            else:
                print("Loading feature")
                feature = self.model.get_feature(img, sess=sess, layer=self.layer)
                features.append(feature)
                self.cache[name] = feature

        self.save_cache_to_file()
        X = np.asarray(features)
        Y = np.asarray(labels)

        return X, Y, names

    def batch_generator(self, dataset: ImageDataset, image_aug: ImagePreprocess, batch_size, shuffle=True):
        X, Y, names = self.load(dataset, image_aug)
        length = len(X)

        while True:
            if shuffle:
                randomize = np.arange(len(X))
                np.random.shuffle(randomize)
                X = X[randomize]
                Y = Y[randomize]

            batch_X = []
            batch_Y = []

            for i in range(length):
                # handle data
                batch_X.append(X[i])
                batch_Y.append(Y[i])
                if len(batch_X) >= batch_size:
                    yield batch_X, batch_Y
                    batch_X = []
                    batch_Y = []

    def get_data(self):
        return self.X, self.Y, self.names


class ImageGenerator(BatchGenerator):

    def __init__(self, file_path: str):
        self.cache = self.read_cache_from_file(file_path)

    def read_cache_from_file(self, file_path):
        return {}

    def load(self, dataset: ImageDataset, image_aug: ImagePreprocess):
        names, imgs, labels = image_aug.apply_dataset(dataset)

        X = np.asarray(imgs)
        Y = np.asarray(labels)

        return X, Y, names

    def batch_generator(self, dataset: ImageDataset, image_aug: ImagePreprocess, batch_size, shuffle=True):
        X, Y, names = self.load(dataset, image_aug)
        length = len(X)

        while True:
            if shuffle:
                randomize = np.arange(len(X))
                np.random.shuffle(randomize)
                X = X[randomize]
                Y = Y[randomize]

            batch_X = []
            batch_Y = []

            for i in range(length):
                # handle data
                batch_X.append(X[i])
                batch_Y.append(Y[i])
                if len(batch_X) >= batch_size:
                    yield batch_X, batch_Y
                    batch_X = []
                    batch_Y = []

    def get_data(self):
        return self.X, self.Y, self.names