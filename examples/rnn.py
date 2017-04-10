import numpy as np
import tensorflow as tf

import tensorflow as tf
from tensorflow.contrib import rnn

from tfwrapper.nets import RNN
from tfwrapper.datasets import mnist

dataset = mnist(size=10000, verbose=True)
X, y, test_X, test_y, _, _ = dataset.getdata(normalize=True, onehot=True, split=True)
with tf.Session() as sess:
	rnn = RNN([28], 28, 128, 10, sess=sess, name='ExampleRNN')
	rnn.train(X, y, epochs=10, sess=sess, verbose=True)