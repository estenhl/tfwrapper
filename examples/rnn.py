import numpy as np
import tensorflow as tf

import tensorflow as tf
from tensorflow.contrib import rnn

from tfwrapper.nets import RNN
from tfwrapper.datasets import mnist

dataset = mnist(size=10000)
dataset = dataset.normalize()
dataset = dataset.onehot()
train, test = dataset.split(0.8)
X = np.squeeze(train.X)
test_X = np.squeeze(test.X)

with tf.Session() as sess:
	rnn = RNN([28], 28, 128, 10, sess=sess, name='ExampleRNN')
	rnn.train(X, train.y, epochs=20, sess=sess)
	_, acc = rnn.validate(test_X, test.y, sess=sess)
	print('Test accuracy: %d%%' % (acc*100))

