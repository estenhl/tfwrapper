import numpy as np
import tensorflow as tf

from tfwrapper import logger

from .cnn import conv2d
from .base import relu
from .base import batch_normalization


def residual_block(*, X=None, modules=3, shortcut=False, filters=[[1, 1], [3, 3], [1, 1]], depths, strides=[1, 1], activation=None, name='residual'):
    if len(filters) == modules:
        pass
    elif len(filters) == 2:
        filters = np.tile(filters, modules)
        filters = np.reshape(filters, (-1, 2))
    else:
        errormsg = 'Invalid filter %s (Must either be of the same length as modules, or a single 1x1 filter)' % repr(filters)
        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)

    if len(strides) == modules:
        pass
    elif len(strides) == 2:
        strides = np.tile(strides, modules)
        strides = np.reshape(strides, (-1, 2))
    else:
        errormsg = 'Invalid strides %s (Must either be of the same length as modules, or a single 1x1 filter)' % repr(strides)
        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)

    if type(depths) is int:
        depths = np.repeat(depths, modules)
    elif len(depths) == modules:
        pass
    else:
        errormsg = 'Invalid depths %s (Must either be of the same length as modules, or a single int)' % repr(depth)
        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)

    if not modules > 1:
        errormsg = 'Invalid number of modules %d for residual_block (Must be >1)' % repr(modules)
        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)

    if X is None:
        return lambda x: residual_block(X=x, modules=modules, shortcut=shortcut, filters=filters, depths=depths, strides=strides, name=name)

    input_X = X
    for i in range(modules - 1):
        X = conv2d(X=X, filter=filters[i], depth=depths[i], strides=list(strides[i]), activation=None, name=name + '/module_%d/conv' % i)
        X = batch_normalization(X=X, name=name + '/module_%d/norm' % i)
        X = relu(X=X, name=name + '/module_%d/relu' % i)

    X = conv2d(X=X, filter=filters[modules - 1], depth=depths[modules - 1], strides=list(strides[modules - 1]), activation=None, name=name + '/module_%d/conv' % (modules - 1))
    X = batch_normalization(X=X, name=name + '/module_%d/norm' % (modules - 1))

    if input_X.get_shape().as_list()[-1] != X.get_shape().as_list()[-1]:
        input_X = conv2d(X=input_X, filter=[1, 1], depth=depths[modules - 1], strides=list(strides[modules - 1]), name=name + '/shortcut')
        input_X = batch_normalization(X=input_X, name=name + '/shortcut/norm')

    X = tf.add(input_X, X, name=name)
    if activation is None:
        return X
    elif activation is 'relu':
        return relu(X=X, name=name)
    
    errormsg = 'Invalid activation %s for residual_block. (Valid is [None, \'relu\'])' % activation
    logger.error(errormsg)
    raise InvalidArgumentException(errormsg)