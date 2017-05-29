import numpy as np
import tensorflow as tf

from tfwrapper import logger

from .cnn import conv2d
from .base import relu


def residual_block(*, input=None, modules=3, shortcut=False, filters=[[1, 1], [3, 3], [1, 1]], depths, strides=[1, 1], activation=None, name='residual'):
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

    if input is None:
        return lambda x: residual_block(input=x, modules=modules, shortcut=shortcut, filters=filters, depths=depths, strides=strides, name=name)

    x = input
    for i in range(modules - 1):
        x = conv2d(input=x, filter=filters[i], depth=depths[i], strides=list(strides[i]), activation=None, name=name + '/conv%d' % i)
        #x = batch_normalize(x)
        x = relu(input=x, name=name + '/relu%d' % i)

    x = conv2d(input=x, filter=filters[modules - 1], depth=depths[modules - 1], strides=list(strides[modules - 1]), activation=None, name=name + '/conv%d' % (modules - 1))
    #x = batch_normalize(x)

    if input.get_shape().as_list()[-1] != x.get_shape().as_list()[-1]:
        input = conv2d(input=input, filter=[1, 1], depth=depths[modules - 1], strides=list(strides[modules - 1]), name=name + '/shortcut')

    x = tf.add(input, x, name=name)
    if activation is None:
        return x
    elif activation is 'relu':
        return relu(input=x, name=name)
    else:
        errormsg = 'Invalid activation %s for residual_block. (Valid is [None, \'relu\'])' % activation