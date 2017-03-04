import pytest
import random
import numpy as np
import tensorflow as tf

from utils.data import onehot
from nets import SingleLayerNeuralNet

"""
def test_single_layer_nn_acc(dims=2, size=50):
	X = np.asarray([[random.randint(0, 1) for j in range(dims)] for i in range(size)])
	y = onehot(np.asarray(list(int(sum(x)/dims) for x in X)))

	with tf.Session() as sess:
		nn = SingleLayerNeuralNet([2], 2, 4, sess=sess)
		nn.learning_rate = 0.01
		nn.train(X, y, epochs=5000, sess=sess)
"""