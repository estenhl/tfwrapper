from .cnn import CNN

class VGG16(CNN):
	def __init__(self):

		layers = [
			self.conv2d(filter=[3, 3], input_depth=3, depth=64),
			self.conv2d(filter=[3, 3], input_depth=64, depth=64),
			self.maxpool2d(k=2),
			self.conv2d(filter=[3, 3], input_depth=64, depth=128),
			self.conv2d(filter=[3, 3], input_depth=128, depth=128),
			self.maxpool2d(k=2),
			self.conv2d(filter=[3, 3], input_depth=128, depth=256),
			self.conv2d(filter=[3, 3], input_depth=256, depth=256),
			self.conv2d(filter=[3, 3], input_depth=256, depth=256),
			self.maxpool2d(k=2),
			self.conv2d(filter=[3, 3], input_depth=256, depth=512),
			self.conv2d(filter=[3, 3], input_depth=512, depth=512),
			self.conv2d(filter=[3, 3], input_depth=512, depth=512),
			self.maxpool2d(k=2),
			self.conv2d(filter=[3, 3], input_depth=512, depth=512),
			self.conv2d(filter=[3, 3], input_depth=512, depth=512),
			self.conv2d(filter=[3, 3], input_depth=512, depth=512),
			self.maxpool2d(k=2))
		]

		raise NotImplementedError('Not finished implementing')
