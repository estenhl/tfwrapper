import inspect

from tfwrapper import logger


def _deprecated_with_replacement(replacement):
    def execute(function):
        def wrapper(*args, **kwargs):
            if len(args) > 0:
                obj = args[0]
                name = obj.__class__.__name__
                if hasattr(obj, replacement):
                    f = getattr(obj, replacement)
                    logger.warning('%s.%s is deprecated! Use %s.%s' % (name, function.__name__, name, replacement))
                    return f(*args[1:], **kwargs)
            logger.warning('%s is deprecated! Use %s' % (function.__name__, replacement))

            frame = inspect.stack()[1]
            module = inspect.getmodule(frame[0])
            if hasattr(module, replacement):
                f = getattr(module, replacement)
                return f(*args, **kwargs)

            logger.warning('%s is deprecated! Replacement function %s does not exist!' % (function.__name__, replacement))
            return function(*args, **kwargs)
        return wrapper
    return execute


def _deprecated(f):
    def wrapper(*args, **kwargs):
        logger.warning('%s is deprecated!' % f.__name__)
        return f(*args, **kwargs)
    return wrapper

def deprecated(arg):
    if type(arg) is str:
        return _deprecated_with_replacement(arg)

    return _deprecated(arg)