import math
import numpy as np
import tensorflow as tf

from tfwrapper import logger


def bias(size, init='zeros', trainable=True, name='bias'):
    return weight([size], init=init, trainable=trainable, name=name)


def weight(shape, init='truncated', trainable=True, name='weight', **kwargs):
    if init == 'truncated':
        stddev = 0.02
        if 'stddev' in kwargs:
            stddev = kwargs['stddev']
        w = tf.truncated_normal(shape, stddev=stddev)
    elif init == 'he_normal':
        # He et al., http://arxiv.org/abs/1502.01852
        fan_in, _ = compute_fan_in_out(shape)
        w = tf.truncated_normal(shape, stddev=math.sqrt(2 / fan_in))
    elif init == 'xavier_normal':
        # Glorot & Bengio, AISTATS 2010 - http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        fan_in, fan_out = compute_fan_in_out(shape)
        w = tf.truncated_normal(shape, stddev=math.sqrt(2 / (fan_in + fan_out)))
    elif init == 'random':
        w = tf.random_normal(shape)
    elif init == 'zeros':
        w = tf.zeros(shape)
    else:
        raise NotImplementedError('Unknown initialization scheme %s' % str(init))

    return tf.Variable(w, trainable=trainable, name=name)


def compute_fan_in_out(weight_shape):
    if len(weight_shape) == 2:
        fan_in = weight_shape[0]
        fan_out = weight_shape[1]
    elif len(weight_shape) in {3, 4, 5}:
        # Assuming convolution kernels (1D, 2D or 3D).
        # TF kernel shape: (..., input_depth, depth)
        receptive_field_size = np.prod(weight_shape[:2])
        fan_in = weight_shape[-2] * receptive_field_size
        fan_out = weight_shape[-1] * receptive_field_size
    else:
        # No specific assumptions.
        fan_in = math.sqrt(np.prod(weight_shape))
        fan_out = math.sqrt(np.prod(weight_shape))
    return fan_in, fan_out

def batch_normalization(X=None, mean=None, variance=None, offset=0, scale=1, name='batch_normalization'):
    if X is None:
        return lambda x: batch_normalization(X=x, mean=mean, variance=variance, offset=offset, scale=scale, name=name)

    _, _, _, depth = X.get_shape().as_list()

    if mean is None and variance is None:
        mean, variance = tf.nn.moments(X, axes=[0])
    elif mean is None:
        mean, _ = tf.nn.moments(X, axes=[0])
    elif variance is None:
        _, variance = tf.nn.moments(X, axes=[0])

    if type(offset) is float or type(offset) is int:
        offset = np.repeat(float(offset), depth)
    elif type(offset) is list or type(offset) is tuple:
        pass
    else:
        errormsg = 'Invalid offset %s. (Valid is [\'int\', \'float\', \'list\', \'tuple\'])' % offset
        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)
    
    if type(scale) is float or type(scale) is int:
        scale = np.repeat(float(scale), depth)
    elif type(scale) is list or type(scale) is tuple:
        pass
    else:
        errormsg = 'Invalid scale %s. (Valid is [\'int\', \'float\', \'list\', \'tuple\'])' % offset
        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)

    beta = tf.Variable(offset, dtype=tf.float32, trainable=True, name=name + '/beta')
    gamma = tf.Variable(scale, dtype=tf.float32, trainable=True, name=name + '/gamma')
    variance_epsilon = tf.Variable(0.0001, trainable=False, name=name + '/epsilon')

    return tf.nn.batch_normalization(X, mean, variance, beta, gamma, variance_epsilon, name=name)


def reshape(shape, name='reshape'):
    return lambda x: tf.reshape(x, shape=shape, name=name)


def out(*, inputs, outputs, init='truncated', trainable=True, name='pred'):
    logger.warning('This layer is an abomination, and should never be used')
    weight_shape = [inputs, outputs]

    def create_layer(x):
        w = weight(weight_shape, init=init, name=name + '/W', trainable=trainable)
        b = bias(outputs, name=name + '/b')
        return tf.add(tf.matmul(x, w), b, name=name)

    return create_layer


def relu(X=None, name='relu'):
    if X is None:
        return lambda x: relu(X=x, name=name)

    return tf.nn.relu(X, name=name)


def softmax(name):
    return lambda x: tf.nn.softmax(x, name=name)
