import os
import numpy as np
import tensorflow as tf

from tfwrapper.nets import ShallowCNN
from tfwrapper.datasets import mnist

from utils import curr_path

def save_model(model, name, sess=None):
	model_path = os.path.join(curr_path, 'data', name)
	if not os.path.isdir(os.path.dirname(model_path)):
		os.mkdir(os.path.dirname(model_path))
	model.save(model_path, sess=sess)

	return model_path

dataset = mnist(size=1000, verbose=True)
X, y, test_X, test_y, _ = dataset.getdata(normalize=True, balance=True, shuffle=True, onehot=True, split=True)
X = np.reshape(X, [-1, 28, 28, 1])

### NO SESSION PROVIDED ###
cnn = ShallowCNN([28, 28, 1], 10, name='NoSessionExample')
cnn.train(X, y, epochs=3, verbose=True)
_, acc = cnn.validate(test_X, test_y)
print('Acc before save without session: %d%%' % (acc * 100))
model_path = save_model(cnn, 'no_session_cnn', sess=None)

loaded_cnn = ShallowCNN([28, 28, 1], 10, name='NoSessionExample')
loaded_cnn.load(model_path)
_, acc = loaded_cnn.validate(test_X, test_y)
print('Acc after load without session: %d%%' % (acc * 100))

tf.reset_default_graph()

### SEPARATE SESSIONS ###
with tf.Session() as sess:
	cnn = ShallowCNN([28, 28, 1], 10, sess=sess, name='SeparateSessionExample')
	cnn.train(X, y, epochs=3, verbose=True, sess=sess)
	_, acc = cnn.validate(test_X, test_y, sess=sess)
	print('Acc before save with separate sessions: %d%%' % (acc * 100))
	model_path = save_model(cnn, 'separate_sessions_cnn', sess=sess)

loaded_cnn = ShallowCNN([28, 28, 1], 10, name='SeparateSessionExample', sess=sess)
tf.reset_default_graph()
with tf.Session() as sess:
	loaded_cnn.load(model_path, sess=sess)
	_, acc = loaded_cnn.validate(test_X, test_y, sess=sess)
	print('Acc after load with separate session: %d%%' % (acc * 100))
