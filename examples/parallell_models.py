import os
import numpy as np
import tensorflow as tf

from tfwrapper.models.nets import RNN
from tfwrapper.models.nets import ShallowCNN
from tfwrapper.datasets import mnist

from utils import curr_path

dataset = mnist(size=5000)
dataset = dataset.normalized()
dataset = dataset.shuffled()
dataset = dataset.onehot_encoded()
train, test = dataset.split(0.8)

X = train.X
test_X = test.X

rnn_X = np.reshape(X, [-1, 28, 28])
rnn_test_X = np.reshape(test_X, [-1, 28, 28])

with tf.Session() as sess:
	cnn = ShallowCNN([28, 28, 1], 10, sess=sess, name='ParallellCNN')
	cnn.train(X, train.y, keep_prob=0.5, epochs=3, sess=sess)
	_, acc = cnn.validate(test_X, test.y, sess=sess)
	print('CNN Acc before save: %d%%' % (acc * 100))

	cnn_path = os.path.join(curr_path, 'data', 'parallell_cnn')
	if not os.path.isdir(os.path.dirname(cnn_path)):
		os.mkdir(os.path.dirname(cnn_path))
	cnn.save(cnn_path, sess=sess)

	rnn = RNN([28], 28, 128, 10, name='ParallellRNN', sess=sess)
	rnn.train(rnn_X, train.y, epochs=10, sess=sess)
	_, acc = rnn.validate(rnn_test_X, test.y, sess=sess)
	print('RNN Acc before save: %d%%' % (acc * 100))
	rnn_path = os.path.join(curr_path, 'data', 'parallell_rnn')
	rnn.save(rnn_path, sess=sess)

tf.reset_default_graph()

with tf.Session() as sess:
	cnn = ShallowCNN([28, 28, 1], 10, name='ParallellCNN', sess=sess)
	cnn.load(cnn_path, sess=sess)
	_, acc = cnn.validate(test_X, test.y, sess=sess)
	print('CNN Acc after load: %d%%' % (acc * 100))

	rnn = RNN([28], 28, 128, 10, name='ParallellRNN', sess=sess)
	rnn.load(rnn_path, sess=sess)
	_, acc = rnn.validate(rnn_test_X, test.y, sess=sess)
	print('RNN Acc after load: %d%%' % (acc * 100))