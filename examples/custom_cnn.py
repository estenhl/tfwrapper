import numpy as np
import tensorflow as tf

from tfwrapper.nets import CNN
from tfwrapper.datasets import mnist

class CustomCNN(CNN):
	learning_rate = 0.002
	batch_size = 512

	def __init__(self, X_shape, classes, sess=None, graph=None, name='CustomCNN'):
		layers = [
			self.reshape([-1, 28, 28, 1], name=name + '_reshape'),
			self.conv2d([5, 5, 1, 32], 32, name=name + '_conv1'),
			self.maxpool2d(k=2, name=name + '_pool1'),
			self.conv2d([5, 5, 32, 64], 64, name=name + '_conv2'),
			self.maxpool2d(k=2, name=name + '_pool2'),
			self.fullyconnected([7*7*64, 512], 512, name=name + '_fc'),
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
		print('Test accuracy: %.2f' % acc)


