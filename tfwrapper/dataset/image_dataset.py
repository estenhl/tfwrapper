import numpy as np

from tfwrapper import logger
from .dataset import Dataset
from .dataset import parse_folder_with_labels_file
from .dataset import parse_datastructure
from .image_loader import ImageLoader


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
        self.paths = dataset.paths

        return dataset.X

    @property
    def y(self):
        if self.loaded_y is not None:
            return self.loaded_y

        dataset, _ = self.next_batch(0, float('inf'))

        self.loaded_X = dataset.X
        self.loaded_y = dataset.y
        self.paths = dataset.paths

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

    def __init__(self, X=np.asarray([]), y=np.asarray([]), paths=None, root_folder=None, labels_file=None, **kwargs):
        if 'loader' in kwargs:
            self._loader = kwargs['loader']
            del kwargs['loader']
        else:
            self._loader = ImageLoader()

        # TODO deprecated, to be removed
        if labels_file is not None and root_folder is not None:
            logger.warning('Do not pass labels_file and root_folder to constructor, use the create classmethods')
            X, y = parse_folder_with_labels_file(root_folder, labels_file)
        elif root_folder is not None:
            logger.warning('Do not pass labels_file and root_folder to constructor, use the create classmethods')
            X, y = parse_datastructure(root_folder)

        # TODO should happen first??
        super().__init__(X=X, y=y, paths=paths, **kwargs)

    @classmethod
    def create_dataset_from_labels_file(cls, root_folder, labels_file):
        X, y = parse_folder_with_labels_file(root_folder, labels_file)
        return ImageDataset(X, y)

    @staticmethod
    def create_dataset_from_folders(root_folder):
        X, y = parse_datastructure(root_folder)
        return ImageDataset(X, y)

    def normalize(self):
        raise NotImplementedError('Unable to normalize an imagedataset which is read on-demand')

    def next_batch(self, cursor, batch_size):
        batch_X = []
        batch_y = []
        paths = []

        while len(batch_X) < batch_size and cursor < len(self):

            path_to_sample = self._X[cursor]
            imgs, names = self.loader.load(path_to_sample, label=self._y[cursor])

            batch_X += imgs
            batch_y += [self._y[cursor]] * len(imgs)
            paths += [{'path': path_to_sample, 'key': n} for n in names]

            cursor += 1

        X = np.asarray(batch_X)
        y = np.asarray(batch_y)

        return Dataset(X=X, y=y, paths=paths), cursor

    def kwargs(self, **kwargs):
        kwargs = super().kwargs(**kwargs)
        kwargs['loader'] = self.loader

        return kwargs