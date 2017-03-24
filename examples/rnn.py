import numpy as np
import tensorflow as tf

import tensorflow as tf
from tensorflow.contrib import rnn

from tfwrapper.datasets import mnist
from tfwrapper.utils.data import batch_data

dataset = mnist(verbose=True)
X, y, test_X, test_y, _ = dataset.getdata(normalize=True, onehot=True, split=True, )

X_batches = batch_data(X, 128)
y_batches = batch_data(y, 128)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
	'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
	'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):
	x = tf.transpose(x, [1, 0, 2])
	x = tf.reshape(x, [-1, n_input])
	x = tf.split(x, n_steps, 0)

	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

	return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	step = 1
	# Keep training until reach max iterations
	for epoch in range(1000):
		for i in range(len(X_batches)):
			batch_x, batch_y = X_batches[i], y_batches[i]
			# Reshape data to get 28 seq of 28 elements
			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
		acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y})
		# Calculate batch loss
		print("Epoch " + str(epoch + 1) + ", Minibatch Loss= " + \
			  "{:.6f}".format(loss) + ", Training Accuracy= " + \
				  "{:.5f}".format(acc))
	print("Optimization Finished!")

	print("Testing Accuracy:", \
		sess.run(accuracy, feed_dict={x: test_X[:500], y: test_y[:500]}))