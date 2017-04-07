import numpy as np
import tensorflow as tf

from tfwrapper import ImageTransformer
from tfwrapper.nets import CNN
from tfwrapper.datasets import mnist


<<<<<<< HEAD
	def __init__(self, X_shape, classes, sess=None, graph=None, name='CustomCNN'):
		layers = [
			self.reshape([-1, 28, 28, 1], name=name + '_reshape'),
			self.conv2d(filter=[5, 5], input_depth=1, depth=32, name=name + '_conv1'),
			self.maxpool2d(k=2, name=name + '_pool1'),
			self.conv2d(filter=[5, 5], input_depth=32, depth=64, name=name + '_conv2'),
			self.maxpool2d(k=2, name=name + '_pool2'),
			self.fullyconnected(input_size=7*7*64, output_size=512, name=name + '_fc'),
			self.out([512, 10], 10, name=name + '_pred')
		]

		super().__init__(X_shape, classes, layers, sess=sess, graph=graph, name=name)


dataset = mnist(size=10000, verbose=True)
X, y, test_X, test_y, _ = dataset.getdata(normalize=True, balance=True, shuffle=True, onehot=True, split=True)
X = np.reshape(X, [-1, 28, 28, 1])

graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		cnn = CustomCNN([28, 28, 1], 10, sess=sess, graph=graph, name='ExampleCustomCNN')
		cnn.train(X, y, epochs=5, sess=sess, verbose=True)
		_, acc = cnn.validate(test_X, test_y, sess=sess)
		print('Test accuracy: %d%%' % (acc*100))
=======
h = 28
w = 28
c = 1

dataset = mnist(size=1000, verbose=True)
transformer = ImageTransformer(rotation_steps=2, max_rotation_angle=15, blur_steps=2, max_blur_sigma=2.5, hflip=True, vflip=True)
X, y, test_X, test_y, _, _ = dataset.getdata(normalize=True, balance=False, shuffle=True, onehot=True,
                                              split=True, translate_labels=True, transformer=transformer)
X = np.reshape(X, [-1, h, w, c])
num_classes = y.shape[1]

name = 'ExampleCustomCNN'
# TODO: make dependent on list of maxpool factors 'k'
twice_reduce = lambda x: -1 * ((-1 * x) // 4)
layers = layers = [
	CNN.reshape([-1, h, w, c], name=name + '/reshape'),
	CNN.conv2d(filter=[5, 5], input_depth=1, depth=32, name=name + '/conv1'),
	CNN.maxpool2d(k=2, name=name + '/pool1'),
	CNN.conv2d(filter=[5, 5], input_depth=32, depth=64, name=name + '/conv2'),
	CNN.maxpool2d(k=2, name=name + '/pool2'),
	CNN.fullyconnected(input_size=twice_reduce(h)*twice_reduce(w)*64, output_size=512, name=name + '/fc'),
	CNN.out([512, num_classes], num_classes, name=name + '/pred')
]
cnn = CNN([h, w, c], num_classes, layers, name=name)
cnn.learning_rate = 1
cnn.train(X, y, epochs=5, verbose=True)
_, acc = cnn.validate(test_X, test_y)
print('Test accuracy: %d%%' % (acc*100))
>>>>>>> master


