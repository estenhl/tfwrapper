import numpy as np
import tensorflow as tf

from tfwrapper import Dataset
from tfwrapper.nets import ShallowCNN
from tfwrapper.datasets import cats_and_dogs


dataset = cats_and_dogs()
X, y, test_X, test_y, _ = dataset.getdata(shuffle=True, translate_labels=True, onehot=True, split=True)

graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		cnn = ShallowCNN([192, 192, 3], 2, sess=sess, graph=graph, name='ExampleShallowCNN')
		cnn.train(X, y, epochs=500, sess=sess, verbose=True)
		_, acc = cnn.validate(test_X, test_y, sess=sess)
		print('Accuracy for example CNN: %.2f' % acc)
