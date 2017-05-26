import numpy as np
import tensorflow as tf

from tfwrapper import logger
from tfwrapper import TFSession
from tfwrapper.nets import VGG16

from .utils import VGG16_NPY_PATH
from .utils import download_vgg16_npy

CHANNEL_MEANS = [103.939, 116.779, 123.68]

class PretrainedVGG16(VGG16):
    def __init__(self, X_shape, *, npy_path=VGG16_NPY_PATH, sess=None, graph=None, name='vgg_16'):
        with TFSession(sess) as sess:
            super().__init__([224, 224, 3], sess=sess, name=name)

            npy_path = download_vgg16_npy(npy_path)
            self.load_from_npy(npy_path, sess=sess)

    def load_from_npy(self, npy_path, sess=None):
        with TFSession(sess, self.graph) as sess:
            data = np.load(npy_path, encoding='latin1').item()
            logger.debug('Loaded vgg16 from %s' % npy_path)
            for key in data:
                root_name = '/'.join([self.name, key])
                weight_name = '/'.join([root_name, 'W'])
                bias_name = '/'.join([root_name, 'b'])

                self.assign_variable_value(weight_name, data[key][0], sess=sess)
                self.assign_variable_value(bias_name, data[key][1], sess=sess)

            logger.debug('Injected variables values into %s' % self.name)