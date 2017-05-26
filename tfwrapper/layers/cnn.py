import tensorflow as tf

from .base import bias
from .base import weight


def conv2d(*, filter, depth, strides=1, padding='SAME', activation='relu', init='truncated', trainable=True, name='conv2d'):
    if len(filter) != 2:
        errormsg = 'conv2d takes filters with exactly 2 dimensions (e.g. [3, 3])'
        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)

    weight_name = name + '/weights'
    bias_name = name + '/biases'

    def create_layer(x):
        input_depth = int(x.get_shape()[-1])

        weight_shape = filter + [input_depth, depth]
        bias_size = depth

        w = weight(weight_shape, name=weight_name, trainable=trainable, init=init)
        b = bias(bias_size, name=bias_name, trainable=trainable)
        conv = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding, name=name)
        conv = tf.nn.bias_add(conv, b)

        if activation == 'relu':
            conv = tf.nn.relu(conv, name=name)
        elif activation == 'softmax':
            conv = tf.nn.softmax(conv, name=name)
        elif activation == 'none':
            pass
        else:
            errormsg = '%s activation is not implemented (Valid: [\'relu\', \'softmax\', \'none\'])' % activation
            logger.error(errormsg)
            raise NotImplementedError(errormsg)

        return conv

    return create_layer


def maxpool2d(*, k=2, strides=2, padding='SAME', name='maxpool2d'):
    return lambda x: tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding=padding, name=name)


def avgpool2d(*, k=2, strides=2, padding='SAME', name='avgpool2d'):
    return lambda x: tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding=padding, name=name)


def flatten(method='avgpool', name='flatten'):
    def create_layer(x):
        _, height, width, _ = x.get_shape()
        filtersize = [1, int(height), int(width), 1]

        if method == 'avgpool':
            return tf.nn.avg_pool(x, ksize=filtersize, strides=filtersize, padding='SAME', name=name)
        elif method == 'maxpool':
            return tf.nn.max_pool(x, ksize=filtersize, strides=filtersize, padding='SAME', name=name)
        else:
            errormsg = '%s method for flatten not impolemented (Valid: [\'avgpool\', \'maxpool\'])' % method
            logger.error(errormsg)
            raise NotImplementedError(errormsg)
    return create_layer