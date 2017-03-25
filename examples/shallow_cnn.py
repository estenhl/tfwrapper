import numpy as np
import tensorflow as tf

from tfwrapper.nets import ShallowCNN
from tfwrapper.datasets import mnist

dataset = mnist(size=10000, verbose=True)
X, y, test_X, test_y, _ = dataset.getdata(normalize=True, balance=True, shuffle=True, onehot=True, split=True)
X = np.reshape(X, [-1, 28, 28, 1])

graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		cnn = ShallowCNN([28, 28, 1], 10, sess=sess, graph=graph, name='ExampleShallowCNN')
		cnn.train(X, y, epochs=5, sess=sess, verbose=True)
		_, acc = cnn.validate(test_X, test_y, sess=sess)
		print('Test accuracy: %d%%' % (acc*100))


