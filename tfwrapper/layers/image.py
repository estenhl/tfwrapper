import random
import tensorflow as tf

from tfwrapper import logger
from tfwrapper.utils.exceptions import InvalidArgumentException


PADDING_CONSTANT = 'CONSTANT'
PADDING_REFLECT = 'REFLECT'
PADDING_SYMMETRIC = 'SYMMETRIC'
_VALID_PADDINGS = [PADDING_CONSTANT, PADDING_REFLECT, PADDING_SYMMETRIC]


def random_crop(*, X=None, padding=(3, 3), ratio=(0.9, 0.9), method=PADDING_CONSTANT, seed=None, name='random_crop'):
    if type(padding) is int:
        padding = (padding, padding)
    elif type(padding) is tuple and len(padding) == 2:
        pass
    else:
        errormsg = 'Invalid padding %s for random_crop. (Must be either an int or a tuple with 2 ints)' % str(padding)
        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)

    if type(ratio) is float:
        ratio = (ratio, ratio)
    elif type(ratio) is tuple and len(ratio) == 2:
        pass
    else:
        errormsg = 'Invalid ratio %s for random_crop. (Must be either a float or a tuple with 2 floats)' % str(ratio)
        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)

    if not method in _VALID_PADDINGS:
        errormsg = 'Invalid method %s for random_crop. (Must be in %s)' % (str(method), str(_VALID_PADDINGS))

    if X is None:
        return lambda x: random_crop(X=x, padding=padding, ratio=ratio, method=method, seed=seed, name=name)

    if seed is not None:
        random.seed(seed)

    paddings = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]]
    paddings = tf.Variable(paddings, dtype=tf.int32, name=name + '/paddings')
    padded = tf.pad(X, paddings, method, name=name + '/pad')

    _, height, width, channels = X.get_shape().as_list()
    scaled_height = int(height * ratio[0])
    scaled_width = int(width * ratio[1])

    y_min = random.randint(0, height - scaled_height)
    x_min = random.randint(0, width - scaled_width)

    begin = [0, y_min, x_min, 0]
    size = [-1, scaled_height, scaled_width, channels]
    cropped = tf.slice(padded, begin, size, name=name + '/crop')

    return resize(X=cropped, img_size=(height, width), name=name)


def resize(*, X=None, img_size, method=tf.image.ResizeMethod.BILINEAR, name='reshape'):
    if X is None:
        return lambda x: resize(X=x, img_size=img_size, method=method, name=name)

    size = tf.Variable(img_size, trainable=False, name=name + '/shape')

    return tf.image.resize_images(X, img_size, method=method)


def normalize_image(name='image_normalization'):
    return lambda x: tf.map_fn(lambda img: tf.image.per_image_standardization(img), x, name=name)


def flip_up_down(seed=None, name='random_flip_ud'):
    return lambda x: tf.map_fn(lambda img: tf.image.random_flip_up_down(img, seed=seed), x, name=name)


def flip_left_right(seed=None, name='random_flip_lr'):
    return lambda x: tf.map_fn(lambda img: tf.image.random_flip_left_right(img, seed=seed), x, name=name)


def brightness(max_delta, seed=None, name='random_brightness'):
    return lambda x: tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta, seed=seed), x, name=name)


def contrast(lower, upper, seed=None, name='random_contrast'):
    return lambda x: tf.map_fn(lambda img: tf.image.random_contrast(img, lower, upper, seed=seed), x, name=name)


def hue(max_delta=0.5, seed=None, name='random_hue'):
    return lambda x: tf.map_fn(lambda img: tf.image.random_hue(img, max_delta, seed=seed), x, name=name)


def saturation(lower, upper, seed=None, name='random_saturation'):
    return lambda x: tf.map_fn(lambda img: tf.image.random_saturation(img, lower, upper, seed=seed), x, name=name)
