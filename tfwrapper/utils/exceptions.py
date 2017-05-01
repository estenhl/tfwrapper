class InvalidArgumentException(ValueError):
	def __init__(self, errormsg):
		super().__init__(errormsg)

class IllegalStateException(ValueError):
	def __init__(self, errormsg):
		super().__init__(errormsg)