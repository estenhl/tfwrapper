from tfwrapper import logger

def raise_exception(errormsg, exception):
    logger.error(errormsg)
    raise exception(errormsg)

class InvalidArgumentException(ValueError):
	def __init__(self, errormsg):
		super().__init__(errormsg)

class IllegalStateException(ValueError):
	def __init__(self, errormsg):
		super().__init__(errormsg)