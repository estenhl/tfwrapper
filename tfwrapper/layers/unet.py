import tensorflow as tf

from .cnn import conv2d


def unet_block(X=None, *, depth, name='unet_block'):
    if X is None:
        return lambda x: unet_block(X=x, depth=depth, name=name)

    X = conv2d(X=X, filter=[3, 3], padding='VALID', name=name + '/conv1')
    X = conv2d(X=X, filter=[3, 3], padding='VALID', name=name + '/conv2')

    return X


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


def deconv2d(X=None, *, filter, depth, name='deconv2d'):
    if X is None:
        return lambda x: deconv2d(X=x, size=size, name=name)

    return X
