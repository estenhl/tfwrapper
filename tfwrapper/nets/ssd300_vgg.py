from .cnn import CNN

class SSD300_VGG(CNN):
	def __init__(self, sess=None, graph=None, name='SSD300_VGG'):
		X_shape = [300, 300, 3]
		classes = 16
		
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
			self.maxpool2d(k=2),
			self.conv2d(filter=[3, 3], input_depth=512, depth=1024),
			self.conv2d(filter=[1, 1], input_depth=1024, depth=1024),
			self.conv2d(filter=[1, 1], input_depth=1024, depth=256, padding='VALID'),
			self.conv2d(filter=[3, 3], input_depth=256, depth=512, padding='VALID'),
			self.conv2d(filter=[1, 1], input_depth=512, depth=128, padding='VALID'),
			self.conv2d(filter=[3, 3], input_depth=128, depth=256, padding='VALID'),
			self.conv2d(filter=[1, 1], input_depth=256, depth=128, padding='VALID'),
			self.conv2d(filter=[3, 3], input_depth=128, depth=256, padding='VALID'),
			self.conv2d(filter=[1, 1], input_depth=256, depth=128, padding='VALID'),
			self.conv2d(filter=[3, 3], input_depth=128, depth=256, padding='VALID'),
		]

		super().__init__(X_shape, classes, layers, sess=sess, graph=graph, name=name)
