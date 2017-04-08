import os
import numpy as np
import tensorflow as tf

from tfwrapper.nets import RNN
from tfwrapper.nets import ShallowCNN
from tfwrapper.datasets import mnist

from utils import curr_path

dataset = mnist(size=10000, verbose=True)
X, y, test_X, test_y, _, _ = dataset.getdata(normalize=True, balance=True, shuffle=True, onehot=True, split=True)
X = np.reshape(X, [-1, 28, 28, 1])
rnn_X = np.reshape(X, [-1, 28, 28])
rnn_test_X = np.reshape(test_X, [-1, 28, 28])

with tf.Session() as sess:
	cnn = ShallowCNN([28, 28, 1], 10, sess=sess, name='ParallellCNN')
	cnn.train(X, y, epochs=3, verbose=True, sess=sess)
	_, acc = cnn.validate(test_X, test_y, sess=sess)
	print('CNN Acc before save: %d%%' % (acc * 100))

	cnn_path = os.path.join(curr_path, 'data', 'parallell_cnn')
	if not os.path.isdir(os.path.dirname(cnn_path)):
		os.mkdir(os.path.dirname(cnn_path))
	cnn.save(cnn_path, sess=sess)

	rnn = RNN([28], 28, 128, 10, name='ParallellRNN')
	rnn.train(rnn_X, y, epochs=10, verbose=True, sess=sess)
	_, acc = rnn.validate(rnn_test_X, test_y, sess=sess)
	print('RNN Acc before save: %d%%' % (acc * 100))
	rnn_path = os.path.join(curr_path, 'data', 'parallell_rnn')
	rnn.save(rnn_path, sess=sess)

tf.reset_default_graph()

with tf.Session() as sess:
	cnn = ShallowCNN([28, 28, 1], 10, name='ParallellCNN', sess=sess)
	cnn.load(cnn_path, sess=sess)
	_, acc = cnn.validate(test_X, test_y, sess=sess)
	print('CNN Acc after load: %d%%' % (acc * 100))

	rnn = RNN([28], 28, 128, 10, name='ParallellRNN')
	rnn.load(rnn_path, sess=sess)
	_, acc = rnn.validate(rnn_test_X, test_y, sess=sess)
	print('RNN Acc after load: %d%%' % (acc * 100))