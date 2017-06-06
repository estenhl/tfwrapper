from tfwrapper import logger
from tfwrapper.utils.exceptions import InvalidArgumentException

def adjust_at_epochs(epochs, values):
    if not len(values) == len(epochs) + 1:
        errormsg = 'len(values) must be exactly len(pivots) + 1'
        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)

    def compute(**kwargs):
        if not 'epoch' in kwargs:
            return values[0]

        epoch = kwargs['epoch']

        if epoch in epochs:
            value = values[epochs.index(epoch) + 1]
            logger.info('Adjusting learning rate to %s' % repr(value))
            return value

        for i in range(len(epochs)):
            if epoch < epochs[i]:
                return values[i]

        return values[-1]

    return compute

def adjust_after_epoch(epoch, *, before, after):
    return adjust_at_epochs([epoch], [before, after])

