import numpy as np
import tensorflow as tf

from tfwrapper import Dataset
from tfwrapper.nets import ShallowCNN
from tfwrapper.datasets import mnist


dataset = mnist(size=5000, verbose=True)
X, y, test_X, test_y, _ = dataset.getdata(shuffle=True, translate_labels=True, onehot=True, split=True)
print('X.shape: ' + str(X.shape))
print('y.shape: ' + str(y.shape))

graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		cnn = ShallowCNN([28, 28, 1], 10, sess=sess, graph=graph, name='ExampleShallowCNN')
		cnn.train(X, y, epochs=10, sess=sess, verbose=True)
		print('Testx: ' + str(test_X.shape))
		print('Testy: ' + str(test_y.shape))

		_, acc = cnn.validate(test_X[100:], test_y[100:], sess=sess)
		print('Accuracy for example CNN: %.2f' % acc)
