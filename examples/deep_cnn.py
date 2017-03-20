import numpy as np
import tensorflow as tf

from tfwrapper.nets import ShallowCNN
from tfwrapper.datasets import cats_and_dogs

dataset = cats_and_dogs(verbose=True)
X, y, test_X, test_y, labels = dataset.getdata(normalize=True, balance=True, shuffle=True, onehot=True, split=True)
X = np.reshape(X, [-1, 196, 196, 3])

graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		cnn = DeepCNN([196, 196, 3], 2, sess=sess, graph=graph, name='ExampleDeepCNN')
		cnn.train(X, y, epochs=5, sess=sess, verbose=True)
		_, acc = cnn.validate(test_X, test_y, sess=sess)
		print('Test accuracy: %.2f' % acc)


