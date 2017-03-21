import numpy as np
import tensorflow as tf

from tfwrapper.datasets import penn_tree_bank
from tfwrapper.utils.data import batch_data

dataset = penn_tree_bank(verbose=True)
X, y, test_X, test_y, _ = dataset.getdata(sequence_length=20, onehot=True, split=True)
X = np.reshape(X, [-1, 20, 1])

x_p = tf.placeholder(tf.float32, [None, 20, 1], name='x_placeholder')
y_p = tf.placeholder(tf.float32, [None, len(dataset.indexes)], name='y_placeholder')

num_hidden = 24
cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

val, state = tf.nn.dynamic_rnn(cell, x_p, dtype=tf.float32)

val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(y_p.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[y_p.get_shape()[1]]))

pred = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(y_p * tf.log(tf.clip_by_value(pred,1e-10,1.0)))

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

correct_pred = tf.not_equal(tf.argmax(y_p, 1), tf.argmax(pred, 1))
cost = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

train_len = len(X) - 128
val_X = X[train_len:]
val_y = y[train_len:]
X = X[:train_len]
y = y[:train_len]
X_batches = batch_data(X, 128)
y_batches = batch_data(y, 128)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(0, 20000):
		for j in range(len(X_batches)):
			sess.run(minimize,{x_p: X_batches[i], y_p: y_batches[i]})
		print("Epoch - " + str(i))
		train_incorrect = sess.run(cost, feed_dict={x_p: X_batches[-2], y_p: y_batches[-2]})
		incorrect = sess.run(cost,{x_p: val_X, y_p: val_y})
		print('Epoch %d train error %.2f val error %.2f' % (i + 1, train_incorrect, 100 * incorrect))

