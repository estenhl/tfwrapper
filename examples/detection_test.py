import os
import tensorflow as tf

from tfwrapper.datasets import cats_and_dogs
from tfwrapper.nets import DeepCNN

dataset = cats_and_dogs(verbose=True)
X, y, test_X, test_y, labels = dataset.getdata(normalize=True, balance=True, translate_labels=True, shuffle=True, onehot=True, split=True)
X_shape = list(X.shape[1:])

graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		cnn = DeepCNN(X_shape, 2, sess=sess, graph=graph, name='CatsAndDogsCNN')
		cnn.train(X, y, epochs=10, sess=sess, verbose=True)
		if not os.path.isdir('models'):
			os.path.mkdir('models')
		if not os.path.isdir('models/detection_test'):
			os.path.mkdir('models/detection_test')
		cnn.save('models/detection_test/cnn', sess=sess)
