import cv2
import pytest
import numpy as np 
import tensorflow as tf
import tflearn.datasets.mnist as mnist

from nets import DeepCNN

def test_deep_cnn_acc():
	X, y, test_X, test_y = mnist.load_data(one_hot=True)

	X_reshaped = []
	for x in X[:5000]:
		X_reshaped.append(cv2.resize(x, (48, 48)))
	X = np.asarray(X_reshaped)

	test_X_reshaped = []
	for x in test_X[:1000]:
		test_X_reshaped.append(cv2.resize(x, (48, 48)))
	test_X = np.asarray(test_X_reshaped)

	print('Finished reshaping')

	with tf.Session() as sess:
		cnn = DeepCNN([48, 48, 1], 10, sess=sess)
		cnn.train(X, y[:5000], epochs=100, sess=sess)