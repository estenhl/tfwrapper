import tensorflow as tf

from .cnn import conv2d
from .base import weight


def unet_block(X=None, *, depth, input_depth=None, name='unet_block'):
    if X is None:
        return lambda x: unet_block(X=x, depth=depth, input_depth=input_depth, name=name)

    X = conv2d(X=X, filter=[3, 3], depth=depth, input_depth=input_depth, padding='VALID', name=name + '/conv1')
    X = conv2d(X=X, filter=[3, 3], depth=depth, padding='VALID', name=name + '/conv2')

    return tf.identity(X, name=name)


def zoom(X=None, *, size, name='zoom'):
    if X is None:
        return lambda x: zoom(X=x, size=size, name=name)

    _, height, width, _ = X.get_shape().as_list()
    slice_height = size[0]
    vertical_start = int((height - slice_height) / 2)

    slice_width = size[1]
    horizontal_start = int((width - slice_width) / 2)

    begin = [0, vertical_start, horizontal_start, 0]
    size = [-1, slice_height, slice_width, -1]

    return tf.slice(X, begin, size, name=name)


def deconv2d(X=None, *, filter, depth, strides=[2, 2], padding='SAME', name='deconv2d'):
    if X is None:
        return lambda x: deconv2d(X=x, filter=filter, depth=depth, name=name)

    channels = X.get_shape()[-1]
    weight_shape = filter + [depth, int(channels)]
    W = weight(weight_shape, name=name + '/W')

    output_shape = tf.stack([tf.shape(X)[0], tf.shape(X)[1] * strides[0], tf.shape(X)[2] * strides[1], depth])
    strides_shape = [1] + strides + [1]

    return tf.nn.conv2d_transpose(X, W, output_shape=output_shape, strides=strides_shape, padding=padding, name=name)
