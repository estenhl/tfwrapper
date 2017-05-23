from tfwrapper import logger

def modify_at_epoch(*, before, after, epoch=150):
    def modify(**kwargs):
        if 'epoch' in kwargs:
            if kwargs['epoch'] == epoch:
                logger.info('Changing learning rate from %f to %f' % (before, after))
                return after
            elif kwargs['epoch'] > epoch:
                return after
                
        return before

    return modify