import os
import numpy as np
import tensorflow as tf

from tfwrapper.nets import ShallowCNN
from tfwrapper.datasets import mnist

from utils import curr_path

dataset = mnist(size=10000, verbose=True)
X, y, test_X, test_y, _ = dataset.getdata(normalize=True, balance=True, shuffle=True, onehot=True, split=True)
X = np.reshape(X, [-1, 28, 28, 1])

with tf.Session() as sess:
	cnn = ShallowCNN([28, 28, 1], 10, sess=sess, name='SeparateSessionExample')
	cnn.train(X, y, epochs=5, verbose=True, sess=sess)
	_, acc = cnn.validate(test_X, test_y, sess=sess)
	print('Acc before save with separate sessions: %d%%' % (acc * 100))

	model_path = os.path.join(curr_path, 'data', 'separate_sessions_cnn')
	if not os.path.isdir(os.path.dirname(model_path)):
		os.mkdir(os.path.dirname(model_path))
	cnn.save(model_path, sess=sess)

tf.reset_default_graph()

with tf.Session() as sess:
	loaded_cnn = ShallowCNN([28, 28, 1], 10, name='SeparateSessionExample', sess=sess)
	loaded_cnn.load(model_path, sess=sess)
	_, acc = loaded_cnn.validate(test_X, test_y, sess=sess)
	print('Acc after load with separate session: %d%%' % (acc * 100))