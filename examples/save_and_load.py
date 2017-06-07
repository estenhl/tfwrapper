import os
import numpy as np
import tensorflow as tf

from tfwrapper.models.nets import ShallowCNN
from tfwrapper.datasets import mnist

from utils import curr_path

dataset = mnist(size=5000)
dataset = dataset.normalize()
dataset = dataset.balance()
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

X = train.X
test_X = test.X

X = np.reshape(X, [-1, 28, 28, 1])
test_X = np.reshape(test_X, [-1, 28, 28, 1])

with tf.Session() as sess:
	cnn = ShallowCNN([28, 28, 1], 10, sess=sess, name='SaveAndLoadExample')
	cnn.train(X, train.y, epochs=5, sess=sess)
	_, acc = cnn.validate(test_X, test.y, sess=sess)
	print('Acc before save: %d%%' % (acc * 100))

	model_path = os.path.join(curr_path, 'data', 'save_and_load_cnn')
	if not os.path.isdir(os.path.dirname(model_path)):
		os.mkdir(os.path.dirname(model_path))
	cnn.save(model_path, sess=sess)

tf.reset_default_graph()

with tf.Session() as sess:
	loaded_cnn = ShallowCNN([28, 28, 1], 10, name='SaveAndLoadExample', sess=sess)
	loaded_cnn.load(model_path, sess=sess)
	_, acc = loaded_cnn.validate(test_X, test.y, sess=sess)
	print('Acc after load: %d%%' % (acc * 100))