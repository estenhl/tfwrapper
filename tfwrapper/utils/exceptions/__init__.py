class InvalidArgumentException(ValueError):
	def __init__(self, errormsg):
		super().__init__(errormsg)