from tfwrapper.layers import conv2d, maxpool2d, fullyconnected, relu, dropout, softmax

from .cnn import CNN

class VGG16(CNN):
	def __init__(self, X_shape, classes=1000, sess=None, name='VGG16'):
		height, width, channels = X_shape
		
		if not ((height % 2 ** 5) == 0 and (width % 2 ** 5) == 0):
			raise ValueError('Height and width must be divisible by %d' % 2 ** 5)
		
		fc_input_size = int(height / (2**5)) * int(width / (2**5)) * 512

		layers = [
			conv2d(filter=[3, 3], depth=64, name=name + '/conv1_1'),
			conv2d(filter=[3, 3], depth=64, name=name + '/conv1_2'),
			maxpool2d(k=2),
			conv2d(filter=[3, 3], depth=128, name=name + '/conv2_1'),
			conv2d(filter=[3, 3], depth=128, name=name + '/conv2_2'),
			maxpool2d(k=2),
			conv2d(filter=[3, 3], depth=256, name=name + '/conv3_1'),
			conv2d(filter=[3, 3], depth=256, name=name + '/conv3_2'),
			conv2d(filter=[3, 3], depth=256, name=name + '/conv3_3'),
			maxpool2d(k=2),
			conv2d(filter=[3, 3], depth=512, name=name + '/conv4_1'),
			conv2d(filter=[3, 3], depth=512, name=name + '/conv4_2'),
			conv2d(filter=[3, 3], depth=512, name=name + '/conv4_3'),
			maxpool2d(k=2),
			conv2d(filter=[3, 3], depth=512, name=name + '/conv5_1'),
			conv2d(filter=[3, 3], depth=512, name=name + '/conv5_2'),
			conv2d(filter=[3, 3], depth=512, name=name + '/conv5_3'),
			maxpool2d(k=2),
			fullyconnected(inputs=fc_input_size, outputs=4096, name=name + '/fc6'),
			relu(name=name + '/relu1'),
			dropout(0.5),
			fullyconnected(inputs=4096, outputs=4096, name=name + '/fc7'),
			relu(name=name + '/relu2'),
			dropout(0.5),
			fullyconnected(inputs=4096, outputs=classes, name=name + '/fc8'),
			softmax(name=name + '/pred')
		]

		super().__init__(X_shape, classes, layers, sess=sess, name=name)

