import numpy as np
import tensorflow as tf

from tfwrapper import logger


def _compute_means(x):
    errormsg = 'Computing channel means is not implemented'
    logger.error(errormsg)
    raise NotImplementedError(errormsg)


def channel_means(means=None, name='channel_means'): 
    def create_layer(x):
        # TODO (28.05.17): Rewrite when pythons stops being stupid with variable name scopes
        if means is None:
            channel_means = _compute_channel_means(x)
        else:
            channel_means = np.asarray(means)

        # TODO (28.05.17): Rewrite in tensorflow (Although tf.tile is weird AF)
        num_repetitions = np.prod(x.get_shape().as_list()[1:-1])
        repeated_means = np.tile(channel_means, num_repetitions)
        original_shape = x.get_shape().as_list()[1:]
        repeated_means = np.reshape(repeated_means, original_shape)

        values = tf.Variable(repeated_means, trainable=False, dtype=tf.float32, name=name + '/values')

        return tf.map_fn(lambda img: tf.subtract(img, values), x, name=name)

    return create_layer


def resize(img_size, method=tf.image.ResizeMethod.BILINEAR, name='reshape'):
    def create_layer(x):
        size = tf.Variable(img_size, trainable=False, name=name + '/shape')

        return tf.image.resize_images(x, img_size, method=method, name=name)

    return create_layer


def normalize_image(name='image_normalization'):
    return lambda x: tf.map_fn(lambda img: tf.image.per_image_standardization(img), x, name=name)


def normalize_batch(name='batch_normalization'):
    raise NotImplementedError('Batch normalization is not implemented')


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
