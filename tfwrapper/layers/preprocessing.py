import tensorflow as tf

from tfwrapper import logger

from .image import random_crop
from .image import flip_left_right
from .image import hue
from .image import contrast
from .image import saturation
from .image import normalize_image

def inception_preprocessing(name=None):
    if name is None:
        logger.warning('Preprocessing layers should be given a name!')
        name = 'InceptionPreprocessing'

    return [
        lambda x: tf.subtract(x, 0.5, name=name + '/subtract'),
        lambda x: tf.multiply(x, 2.0, name=name + '/multiply')
    ]

def vgg_preprocessing(name=None):
    if name is None:
        logger.warning('Preprocessing layers should be given a name!')
        name = 'VGGPreprocessing'

    # TODO (01.06.17): Add centralized cropping
def randomized_preprocessing(normalize=True, name=None):
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
        layers += normalize_image(name=name + '/normalize_image')

    return layers