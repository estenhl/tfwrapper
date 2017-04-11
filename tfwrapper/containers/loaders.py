import numpy as np
import tensorflow as tf
from tfwrapper import Dataset
from tfwrapper.containers import ImageContainer
from tfwrapper.containers import features
from tfwrapper.containers.image_augment import ImagePreprocess
from tfwrapper.nets.pretrained.pretrained_model import PretrainedModel


class CachedFeatureLoader():

    def __init__(self, file_path: str, model: PretrainedModel, layer=None):
        self.file_path = file_path
        self.model = model
        self.layer = layer

        self.cache = self.read_cache_from_file()

    def read_cache_from_file(self):
        return features.parse_features(self.file_path)

    def save_cache_to_file(self):
        features.write_features(self.cache, self.file_path)

    def create_dataset(self, dataset: ImageContainer, image_aug: ImagePreprocess):
        names, imgs, labels = image_aug.apply_dataset(dataset)

        features = []
        sess = tf.Session(graph=self.model.graph)

        counter_log_interval = len(names)/10
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

        return Dataset(X, Y), names


    def batch_generator(self, dataset: ImageContainer, image_aug: ImagePreprocess, batch_size, shuffle=True):
        dataset, names = self.create_dataset(dataset, image_aug)
        length = len(dataset.X)

        while True:
            if shuffle:
                dataset = dataset.shuffle()

            batch_X = []
            batch_Y = []

            for i in range(length):
                # handle data
                batch_X.append(dataset.X[i])
                batch_Y.append(dataset.Y[i])
                if len(batch_X) >= batch_size:
                    yield batch_X, batch_Y
                    batch_X = []
                    batch_Y = []


class ImageLoader():

    def create_dataset(self, dataset: ImageContainer, image_aug: ImagePreprocess):
        names, imgs, labels = image_aug.apply_dataset(dataset)

        X = np.asarray(imgs)
        Y = np.asarray(labels)

        return Dataset(X, Y), names

    def batch_generator(self, dataset: ImageContainer, image_aug: ImagePreprocess, batch_size, shuffle=True):
        dataset, names = self.create_dataset(dataset, image_aug)
        length = len(dataset.X)

        while True:
            if shuffle:
                dataset = dataset.shuffle()
            batch_X = []
            batch_Y = []

            #Can use dataset batcher in future?
            for i in range(length):
                # handle data
                batch_X.append(dataset.X[i])
                batch_Y.append(dataset.Y[i])
                if len(batch_X) >= batch_size:
                    yield batch_X, batch_Y
                    batch_X = []
                    batch_Y = []
