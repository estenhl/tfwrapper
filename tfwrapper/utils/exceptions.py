from tfwrapper import logger
from tfwrapper.utils.decorators import deprecated


@deprecated
def raise_exception(errormsg, exception):
    logger.error(errormsg)
    raise exception(errormsg)


def log_and_raise(exception, errormsg):
	logger.error(errormsg)
	raise exception(errormsg)


class InvalidArgumentException(ValueError):
	def __init__(self, errormsg):
		super().__init__(errormsg)


class IllegalStateException(ValueError):
	def __init__(self, errormsg):
		super().__init__(errormsg)