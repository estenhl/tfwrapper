import math
import numpy as np
import tensorflow as tf


def bias(size, init='zeros', trainable=True, name='bias'):
    return weight([size], init=init, trainable=trainable, name=name)


def weight(shape, init='truncated', stddev=0.02, trainable=True, name='weight'):
    if init == 'truncated':
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

def batch_normalization(input=None, mean=None, variance=None, offset=None, scale=None, name='batch_normalization'):
    if input is None:
        return lambda x: batch_normalization(input=x, mean=mean, variance=variance, offset=offset, scale=scale, name=name)

    if mean is None and variance is None:
        mean, variance = tf.nn.moments(input, axes=[0])
    elif mean is None:
        mean, _ = tf.nn.moments(input, axes=[0])
    elif variance is None:
        _, variance = tf.nn.moments(input, axes=[0])

    variance_epsilon = 0.0001
    return tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon, name=name)


def reshape(shape, name='reshape'):
    return lambda x: tf.reshape(x, shape=shape, name=name)


def out(*, inputs, outputs, init='truncated', trainable=True, name='pred'):
    weight_shape = [inputs, outputs]

    def create_layer(x):
        w = weight(weight_shape, init=init, name=name + '/W', trainable=trainable)
        b = bias(outputs, name=name + '/b')
        return tf.add(tf.matmul(x, w), b, name=name)

    return create_layer


def relu(input=None, name='relu'):
    if input is None:
        return lambda x: relu(input=x, name=name)

    return tf.nn.relu(input, name=name)


def softmax(name):
    return lambda x: tf.nn.softmax(x, name=name)
