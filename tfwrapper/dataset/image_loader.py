import os

from tfwrapper import twimage
from .image_preprocessor import ImagePreprocessor


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