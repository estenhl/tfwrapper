from tfwrapper import logger

def deprecated(overriding_func):
    """
    Created a deprecated decorator with an argument, which is the new function that should be called.
    If more than one non-keyword argument is passed, the first is treated as self, and the overriding
    function is called from self
    """
    def deprecated_exec(function):
        def wrapper(*args, **kwargs):
            logger.warning('%s is deprecated! Use %s' % (function.__name__, overriding_func))
            if len(args) > 0:
                self = args[0]
                if hasattr(self, overriding_func):
                    new_func = getattr(self, overriding_func)
                    return new_func(*args[1:], **kwargs)
            return function(*args, **kwargs)
        return wrapper
    return deprecated_exec