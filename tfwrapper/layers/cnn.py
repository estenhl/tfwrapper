import numpy as np
import tensorflow as tf

from tfwrapper import logger

from .base import bias
from .base import weight


def conv2d(*, X=None, filter, depth=None, strides=1, padding='SAME', activation='relu', init='truncated', trainable=True, name='conv2d'):
    if len(filter) != 2:
        errormsg = 'conv2d takes filters with exactly 2 dimensions (e.g. [3, 3])'
        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)

    if type(strides) is int:
        strides = [1, strides, strides, 1]
    elif type(strides) is list:
        if len(strides) == 2:
            strides = [1] + strides + [1]
        elif type(strides) is list and len(strides) == 4:
            pass
        else:
            errormsg = 'Invalid strides %s. (Must be a single integer, a list with 2 elements or a list with 4 elements)' % strides
            logger.error(errormsg)
            raise InvalidArgumentException(errormsg)
    else:
        errormsg = 'Invalid strides %s. (Must be a single integer, a list with 2 elements or a list with 4 elements)' % strides
        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)

    if X is None:
        return lambda x: conv2d(X=x, filter=filter, depth=depth, strides=strides, padding=padding, activation=activation, init=init, trainable=trainable, name=name)

    weight_name = name + '/W'
    bias_name = name + '/b'

    input_depth = int(X.get_shape()[-1])
    if depth is None:
        depth = input_depth

    weight_shape = filter + [input_depth, depth]
    bias_size = depth

    w = weight(weight_shape, name=weight_name, trainable=trainable, init=init)
    b = bias(bias_size, name=bias_name, trainable=trainable)
    conv = tf.nn.conv2d(X, w, strides=strides, padding=padding, name=name)
    conv = tf.nn.bias_add(conv, b)

    if activation is 'relu':
        return tf.nn.relu(conv, name=name)
    elif activation is 'softmax':
        return tf.nn.softmax(conv, name=name)
    elif activation is None:
        return conv
    else:
        errormsg = '%s activation is not implemented (Valid: [\'relu\', \'softmax\', \'none\'])' % activation
        logger.error(errormsg)
        raise NotImplementedError(errormsg)


def maxpool2d(X=None, k=2, strides=2, padding='SAME', name='maxpool2d'):
    if X is None:
        return lambda x: maxpool2d(X=x, k=k, strides=strides, padding=padding, name=name)

    return tf.nn.max_pool(X, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding=padding, name=name)


def avgpool2d(X=None, k=2, strides=2, padding='SAME', name='avgpool2d'):
    if X is None:
        return lambda x: avgpool2d(X=x, k=k, strides=strides, padding=padding, name=name)

    return tf.nn.avg_pool(X, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding=padding, name=name)


def flatten(input=None, method='avgpool', name='flatten'):
    if input is None:
        return lambda x: flatten(input=x, method=method, name=name)

    _, height, width, _ = input.get_shape()
    filtersize = [1, int(height), int(width), 1]

    if method == 'avgpool':
        return tf.nn.avg_pool(input, ksize=filtersize, strides=filtersize, padding='SAME', name=name)
    elif method == 'maxpool':
        return tf.nn.max_pool(input, ksize=filtersize, strides=filtersize, padding='SAME', name=name)
    else:
        errormsg = '%s method for flatten not implemented (Valid: [\'avgpool\', \'maxpool\'])' % method
        logger.error(errormsg)
        raise NotImplementedError(errormsg)
