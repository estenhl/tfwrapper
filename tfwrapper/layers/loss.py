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


def multiclass_hinge(y, preds, name, method=MULTICLASS_HINGE_CRAMMER_SINGER):
    batch_size, length = preds.get_shape()
    shape = (length,)

    onehot = tf.one_hot(y, length, name=name + '/onehot')
    reverse_onehot = tf.subtract(tf.ones((batch_size, length)), onehot, name=name + '/reverse_onehot')

    w_y = tf.multiply(onehot, preds, name=name + '/w_y/multiply')
    w_y = tf.map_fn(lambda x: tf.reduce_sum(x), w_y, name=name + '/w_y/reduce')
    
    w_t = tf.multiply(reverse_onehot, preds, name=name + '/w_y/multiply')
    w_t = tf.map_fn(lambda x: tf.reduce_max(x), w_t, name=name + '/w_y/reduce')

    diff = tf.subtract(w_t, w_y, name=name + '/individual_losses/diff')
    add = tf.add(tf.ones(shape), diff, name=name + '/add')
    floored = tf.maximum(tf.zeros(shape), add, name=name + '/floored')

    return tf.reduce_mean(floored, name=name)


def hinge(y, preds, num_classes=None, name='hinge', **kwargs):
    if num_classes is not 2:
        return multiclass_hinge(y, preds, name, **kwargs)

    length = y.get_shape()[0]
    shape = (length,)

    if length != preds.get_shape()[0]:
        raise_exception('Hinge loss requires y and predictions to be of the same length', InvalidArgumentException)

    multiplied = tf.multiply(y, preds, name=name + '/multiply')
    subtracted = tf.subtract(tf.ones(shape), multiplied, name=name + '/subtracted')
    floored = tf.maximum(tf.zeros(shape), subtracted, name=name + '/floored')

    return tf.reduce_mean(floored, name=name)


def squared_hinge(y, preds, num_classes=None, name='squared_hinge'):
    return tf.square(hinge(y, preds, num_classes=num_classes, name=name + '/provisional'), name=name)




