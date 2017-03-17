import numpy as np
import tensorflow as tf

from tfwrapper.nets import ShallowCNN
from tfwrapper.datasets import mnist

dataset = mnist(size=4000)
X, y, test_X, test_y, _ = dataset.getdata(normalize=True, balance=True, shuffle=True, onehot=True, split=True)

graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		cnn = ShallowCNN([28, 28, 1], 10, sess=sess, graph=graph, name='ExampleShallowCNN')
		cnn.learning_rate = 0.1
		cnn.train(X, y, sess=sess, verbose=True)