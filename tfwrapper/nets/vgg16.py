from .cnn import CNN

class VGG16(CNN):
	def __init__(self, X_shape, classes=1000, sess=None, name='VGG16'):
		height, width, channels = X_shape
		
		if not ((height % 2 ** 5) == 0 and (width % 2 ** 5) == 0):
			raise ValueError('Height and width must be divisible by %d' % 2 ** 5)
		
		fc_input_size = int(height / (2**5)) * int(width / (2**5)) * 512

		layers = [
			self.conv2d(filter=[3, 3], depth=64, name=name + '/conv1/conv1_1'),
			self.conv2d(filter=[3, 3], depth=64, name=name + '/conv1/conv1_2'),
			self.maxpool2d(k=2),
			self.conv2d(filter=[3, 3], depth=128, name=name + '/conv2/conv2_1'),
			self.conv2d(filter=[3, 3], depth=128, name=name + '/conv2/conv2_2'),
			self.maxpool2d(k=2),
			self.conv2d(filter=[3, 3], depth=256, name=name + '/conv3/conv3_1'),
			self.conv2d(filter=[3, 3], depth=256, name=name + '/conv3/conv3_2'),
			self.conv2d(filter=[3, 3], depth=256, name=name + '/conv3/conv3_3'),
			self.maxpool2d(k=2),
			self.conv2d(filter=[3, 3], depth=512, name=name + '/conv4/conv4_1'),
			self.conv2d(filter=[3, 3], depth=512, name=name + '/conv4/conv4_2'),
			self.conv2d(filter=[3, 3], depth=512, name=name + '/conv4/conv4_3'),
			self.maxpool2d(k=2),
			self.conv2d(filter=[3, 3], depth=512, name=name + '/conv5/conv5_1'),
			self.conv2d(filter=[3, 3], depth=512, name=name + '/conv5/conv5_2'),
			self.conv2d(filter=[3, 3], depth=512, name=name + '/conv5/conv5_3'),
			self.maxpool2d(k=2),
			self.fullyconnected(inputs=fc_input_size, outputs=4096, name=name + '/fc6'),
			self.relu(name=name + '/relu1'),
			self.dropout(0.5),
			self.fullyconnected(inputs=4096, outputs=4096, name=name + '/fc7'),
			self.relu(name=name + '/relu2'),
			self.dropout(0.5),
			self.fullyconnected(inputs=4096, outputs=classes, name=name + '/fc8'),
			self.softmax(name=name + '/pred')
		]

		super().__init__(X_shape, classes, layers, sess=sess, name=name)
