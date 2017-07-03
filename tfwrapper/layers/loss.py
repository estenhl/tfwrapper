import tensorflow as tf

from tfwrapper.utils.exceptions import raise_exception
from tfwrapper.utils.exceptions import InvalidArgumentException

def mse(y, preds, name='mse'):
    if y.get_shape()[0] != preds.get_shape()[0]:
        raise_exception('MSE loss requires y and predictions to be of the same length', InvalidArgumentException)
    
    squared = tf.square(preds - y, name=name + '/squared')

    return tf.reduce_mean(squared, name=name)


# http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf
MULTICLASS_HINGE_CRAMMER_SINGER = 'crammer_singer'


def multiclass_hinge(y, preds, method=MULTICLASS_HINGE_CRAMMER_SINGER, delta=10e-8, name='hinge'):
    ones = tf.ones(tf.shape(y))
    reverse_onehot = tf.subtract(ones, y, name=name + '/reverse_onehot')

    w_y = tf.multiply(y, preds, name=name + '/w_y/multiply')
    w_y = tf.map_fn(lambda x: tf.reduce_sum(x), w_y, name=name + '/w_y/reduce')
    
    w_t = tf.multiply(reverse_onehot, preds, name=name + '/w_y/multiply')
    w_t = tf.map_fn(lambda x: tf.reduce_max(x), w_t, name=name + '/w_y/reduce')

    diff = tf.subtract(w_t + delta, w_y, name=name + '/individual_losses/diff')
    flat_ones = tf.ones(tf.shape(diff), name=name + '/flat_ones')
    add = tf.add(flat_ones, diff, name=name + '/add')
    flat_zeros = tf.zeros(tf.shape(add), name=name + '/flat_zeros')
    floored = tf.maximum(flat_zeros, add, name=name + '/floored')

    return tf.reduce_mean(floored, name=name)


def squared_multiclass_hinge(y, preds, method=MULTICLASS_HINGE_CRAMMER_SINGER, delta=10e-8, name='hinge'):
    return tf.square(multiclass_hinge(y, preds, method=method, delta=delta, name=name + '/provisional'), name=name)


def binary_hinge(y, preds, name='hinge'):
    # Transform [0, 1] y matrix to [-1, 1]
    # TODO (03.07.17): Should check that values are actually in [0, 1]
    scaled = tf.multiply(y, 2.0, name=name + '/scaled')
    shifted = tf.subtract(scaled, 1.0, name=name + '/shifted')

    # Calculate binary hinge loss
    multiplied = tf.multiply(shifted, preds, name=name + '/multiply')
    ones = tf.map_fn(lambda x: 1.0, y, name=name + '/ones')
    zeros = tf.subtract(ones, 1.0, name=name + '/zeros')
    subtracted = tf.subtract(ones, multiplied, name=name + '/subtracted')
    floored = tf.maximum(zeros, subtracted, name=name + '/floored')

    return tf.reduce_mean(floored, name=name)


def squared_binary_hinge(y, preds, num_classes=None, name='squared_hinge'):
    return tf.square(binary_hinge(y, preds, name=name + '/provisional'), name=name)




