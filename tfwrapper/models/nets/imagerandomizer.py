import random
import tensorflow as tf

from tfwrapper import logger
from tfwrapper import TFSession
from tfwrapper.layers import flip_up_down, flip_left_right, brightness, hue, contrast, saturation
from tfwrapper.utils.exceptions import InvalidArgumentException

from .cnn import CNN

class ImageRandomizer(CNN):
    def __init__(self, X_shape, seed=None, flip_ud=False, flip_lr=False, brightness_delta=None, hue_delta=None, contrast_range=None, saturation_range=None, shuffle=True, name='ImageRandomizer', sess=None):
        layers = []

        if flip_ud:
            layers.append(flip_ud_down(seed=seed, name=name + '/random_flip_ud'))

        if flip_lr:
            layers.append(flip_left_right(seed=seed, name=name + '/random_flip_lr'))

        if brightness_delta is not None:
            layers.append(brightness(brightness_delta, seed=seed, name=name + '/random_brightness'))

        if hue_delta is not None:
            layers.append(hue(hue_delta, seed=seed, name=name + '/random_hue'))

        if contrast_range is not None:
            if len(contrast_range) == 2 and isinstance(contrast_range[0], (int, float)) and isinstance(contrast_range[1], (int, float)):
                layers.append(contrast(contrast_range[0], contrast_range[1], seed=seed, name=name + '/random_contrast'))
            else:
                errormsg = 'Contrast_range must be [number, number]'
                logger.error(errormsg)
                raise InvalidArgumentException(errormsg)

        if saturation_range is not None:
            if len(saturation_range) and isinstance(saturation_range[0], (int, float)) == isinstance(saturation_range[1], (int, float)):
                layers.append(saturation(saturation_range[0], saturation_range[1], seed=seed, name=name + '/random_saturation'))
            else:
                errormsg = 'Saturation_range must be [float, float]'
                logger.error(errormsg)
                raise InvalidArgumentException(errormsg)

        if shuffle:
            random.shuffle(layers)

        with TFSession(sess) as sess:
            super().from_shape(X_shape, X_shape, layers, sess=sess, name=name)

    def loss_function(self):
        return tf.Variable(1)

    def optimizer_function(self):
        return tf.Variable(1)

    def accuracy_function(self):
        return tf.Variable(1)