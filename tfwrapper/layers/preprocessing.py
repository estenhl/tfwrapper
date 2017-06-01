import tensorflow as tf

from tfwrapper import logger

from .image import hue
from .image import contrast
from .image import saturation
from .image import random_crop
from .image import channel_means
from .image import flip_left_right
from .image import normalize_image


# https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3, lines 59-61
VGG_CHANNEL_MEANS = [103.939, 116.779, 123.68]


def inception_preprocessing(name=None):
    logger.warning('Using Inception preprocessing for EVALUATION (as opposed to TRAINING)')
    return inception_eval_preprocessing(name=name)


# https://github.com/tensorflow/models/blob/master/slim/preprocessing/inception_preprocessing.py
def inception_eval_preprocessing(name=None):
    if name is None:
        logger.warning('Preprocessing layers should be given a name!')
        name = 'InceptionPreprocessing'

    return [
        lambda x: tf.subtract(x, 0.5, name=name + '/subtract'),
        lambda x: tf.multiply(x, 2.0, name=name + '/multiply')
    ]


def vgg_preprocessing(means=VGG_CHANNEL_MEANS, name=None):
    logger.warning('Using VGG preprocessing for EVALUATION (as opposed to TRAINING)')
    return vgg_eval_preprocessing(means=means, name=name)


# https://github.com/tensorflow/models/blob/master/slim/preprocessing/vgg_preprocessing.py
def vgg_eval_preprocessing(means=VGG_CHANNEL_MEANS, name=None):
    if name is None:
        logger.warning('Preprocessing layers should be given a name!')
        name = 'VGGPreprocessing'

    # TODO (01.06.17): Add centralized cropping
    return [channel_means(means=means, name=name + '/channel_means')]


def randomized_preprocessing(normalize=True, seed=None, name=None):
    if name is None:
        logger.warning('Preprocessing layers should be given a name!')
        name = 'RandomizedPreprocessing'

    layers = [
        random_crop(padding=3, ratio=0.85, seed=seed, name=name + '/random_crop'),
        flip_left_right(seed=seed, name=name + '/flip_lr'),
        hue(0.2, seed=seed, name=name + '/hue'),
        contrast(0.7, 1, seed=seed, name=name + '/contrast'),
        saturation(0.7, 1.3, seed=seed, name=name + '/saturation')
    ]

    if normalize:
        layers.append(normalize_image(name=name + '/normalize_image'))

    return layers