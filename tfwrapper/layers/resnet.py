import numpy as np

from tfwrapper import logger

from .cnn import conv2d

def residual_block(*, length=3, shortcut=False, filters, depth, strides=[1, 1], name='residual'):
    if len(filters) == length:
        pass
    elif len(filters) == 2:
        filters = np.tile(filters, length)
        filters = np.reshape(filters, (-1, 2))
    else:
        errormsg = 'Invalid filter %s (Must either be of the same length as length, or a single 1x1 filter)' % repr(filters)
        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)

    if len(strides) == length:
        pass
    elif len(strides) == 2:
        strides = np.tile(strides, length)
        strides = np.reshape(strides, (-1, 2))
    else:
        errormsg = 'Invalid strides %s (Must either be of the same length as length, or a single 1x1 filter)' % repr(strides)
        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)
