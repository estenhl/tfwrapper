import numpy as np
import tensorflow as tf

from tfwrapper.nets import CNN
from tfwrapper.datasets import mnist

h = 28
w = 28
c = 1

dataset = mnist(size=10000, verbose=True)
X, y, test_X, test_y, _ = dataset.getdata(normalize=True, balance=True, shuffle=True, onehot=True, split=True)
X = np.reshape(X, [-1, h, w, c])
num_classes = y.shape[1]

graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		name = 'ExampleCustomCNN'
		# TODO: make dependent on list of maxpool factors 'k'
		twice_reduce = lambda x: -1 * ((-1 * x) // 4)
		layers = layers = [
			CNN.reshape([-1, h, w, c], name=name + '_reshape'),
			CNN.conv2d(filter=[5, 5], input_depth=1, depth=32, name=name + '_conv1'),
			CNN.maxpool2d(k=2, name=name + '_pool1'),
			CNN.conv2d(filter=[5, 5], input_depth=32, depth=64, name=name + '_conv2'),
			CNN.maxpool2d(k=2, name=name + '_pool2'),
			CNN.fullyconnected(input_size=twice_reduce(h)*twice_reduce(w)*64, output_size=512, name=name + '_fc'),
			CNN.out([512, num_classes], num_classes, name=name + '_pred')
		]
		cnn = CNN([h, w, c], num_classes, layers, sess=sess, graph=graph, name=name)
		cnn.train(X, y, epochs=5, sess=sess, verbose=True)
		_, acc = cnn.validate(test_X, test_y, sess=sess)
		print('Test accuracy: %d%%' % (acc*100))


