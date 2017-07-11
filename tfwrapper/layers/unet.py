import tensorflow as tf

from .cnn import conv2d
from .base import weight


def unet_block(X=None, *, depth, name='unet_block'):
    if X is None:
        return lambda x: unet_block(X=x, depth=depth, name=name)

    X = conv2d(X=X, filter=[3, 3], depth=depth, padding='VALID', name=name + '/conv1')
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

    batch_size, height, width, channels = X.get_shape()
    weight_shape = filter + [depth, int(channels)]
    W = weight(weight_shape, name=name + '/W')

    output_shape = [int(batch_size), int(height * 2), int(width * 2), depth]
    strides_shape = [1] + strides + [1]

    print('X: ' + str(X))
    print('W: ' + str(W))
    print('Output_shape: ' + str(output_shape))

    return tf.nn.conv2d_transpose(X, W, output_shape, strides=strides_shape, padding=padding, name=name)
