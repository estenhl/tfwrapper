import numpy as np
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper import SupervisedModel

class NeuralNet(SupervisedModel):
	batch_size = 10
	def __init__(self, X_shape, classes, layers, sess=None, graph=None, name='NeuralNet'):

		self.X_shape = X_shape
		self.classes = classes
		self.y_size = classes
		self.name = name
		#self.input_size = np.prod(X_shape)
		#self.output_size = y_size

		self.X = tf.placeholder(tf.float32, [None] + X_shape, name=self.name + '_X_placeholder')
		self.y = tf.placeholder(tf.float32, [None, classes], name=self.name + '_y_placeholder')

		prev = self.X
		for layer in layers:
			prev = layer(prev)
		self.pred = prev

		print('Loss: ' + str(sess))
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y, name=self.name + '_softmax'), name=self.name + '_loss')
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name=self.name + '_adam').minimize(self.loss, name=self.name + '_optimizer')
		self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1), name=name + '_correct_pred')
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name=self.name + '_accuracy')

		self.graph = graph

	def loss_function(self):
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y, name=self.name + '_softmax'), name=self.name + '_loss')

	def optimizer_function(self):
		return tf.train.AdamOptimizer(learning_rate=self.learning_rate, name=self.name + '_adam').minimize(self.loss, name=self.name + '_optimizer')

	def accuracy_function(self):
		return tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name=self.name + '_accuracy')

	def fullyconnected(self, prev, weight, bias, name=None):
		fc = tf.reshape(prev, [-1, weight.get_shape().as_list()[0]], name=name + '_reshape')
		fc = tf.add(tf.matmul(fc, weight), bias, name=name + '_add')
		fc = tf.nn.relu(fc, name=name)

		return fc

	def train(self, X, y, val_X=None, val_y=None, validate=True, epochs=5000, sess=None, verbose=False):
		print('TRAINING NEURAL NET')
		assert len(X) == len(y)

		X = np.reshape(X, [-1, 64, 64, 3])
		y = np.reshape(y, [-1, self.y_size])
		if val_X is None and validate:
			X, y, val_X, val_y = split_dataset(X, y)
			
		X_batches = self.batch_data(X)
		y_batches = self.batch_data(y)
		num_batches = len(X_batches)
		
		# Parameters
		learning_rate = 0.001
		training_iters = 200000
		batch_size = 10
		display_step = 10

		# Network Parameters
		n_classes = 2 # MNIST total classes (0-9 digits)
		dropout = 0.75 # Dropout, probability to keep units

		# tf Graph input
		x = tf.placeholder(tf.float32, [None, 64, 64, 3])
		y_ = tf.placeholder(tf.float32, [None, n_classes])
		keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


		# Create some wrappers for simplicity
		def conv2d(x, W, b, strides=1):
			# Conv2D wrapper, with bias and relu activation
			x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
			x = tf.nn.bias_add(x, b)
			return tf.nn.relu(x)


		def maxpool2d(x, k=2):
			# MaxPool2D wrapper
			return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
								  padding='SAME')


		# Create model
		def conv_net(x, weights, biases, dropout):
			# Reshape input picture
			x = tf.reshape(x, shape=[-1, 64, 64, 3])

			# Convolution Layer
			conv1 = conv2d(x, weights['wc1'], biases['bc1'])
			# Max Pooling (down-sampling)
			conv1 = maxpool2d(conv1, k=2)

			# Convolution Layer
			conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
			# Max Pooling (down-sampling)
			conv2 = maxpool2d(conv2, k=2)

			conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
			conv3 = maxpool2d(conv3, k=2)

			# Fully connected layer
			# Reshape conv2 output to fit fully connected layer input
			fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
			fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
			fc1 = tf.nn.relu(fc1)
			# Apply Dropout
			fc1 = tf.nn.dropout(fc1, dropout)

			# Output, class prediction
			out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
			return out

		# Store layers weight & bias
		weights = {
			# 5x5 conv, 1 input, 32 outputs
			'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
			# 5x5 conv, 32 inputs, 64 outputs
			'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
			'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
			# fully connected, 7*7*64 inputs, 1024 outputs
			'wd1': tf.Variable(tf.random_normal([8*8*128, 1024])),
			# 1024 inputs, 10 outputs (class prediction)
			'out': tf.Variable(tf.random_normal([1024, n_classes]))
		}

		biases = {
			'bc1': tf.Variable(tf.random_normal([32])),
			'bc2': tf.Variable(tf.random_normal([64])),
			'bc3': tf.Variable(tf.random_normal([128])),
			'bd1': tf.Variable(tf.random_normal([1024])),
			'out': tf.Variable(tf.random_normal([n_classes]))
		}

		# Construct model
		pred = conv_net(x, weights, biases, keep_prob)

		# Define loss and optimizer
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

		# Evaluate model
		correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		# Initializing the variables
		init = tf.global_variables_initializer()

		from collections import Counter
		print('Counts: ' + str(Counter([str(x) for x in y])))
		# Launch the graph
		with tf.Session() as sess:
			sess.run(init)
			# Keep training until reach max iterations
			for epoch in range(500):
				for i in range(num_batches):
					batch_x, batch_y = X_batches[i], y_batches[i]
					# Run optimization op (backprop)
					sess.run(optimizer, feed_dict={x: batch_x, y_: batch_y,
												   keep_prob: dropout})
					# Calculate batch loss and accuracy
				loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
																y_: batch_y,
																keep_prob: 1.})
				print('Epoch %d: loss: %.2f, acc %.2f' % (epoch + 1, loss, acc))
				preds = sess.run(pred, feed_dict={x: batch_x, keep_prob: 1.})
				for i in range(0, len(preds)):
					print(str([round(x, 2) for x in preds[i]]) + ': ' + str(batch_y[i]))
			print("Optimization Finished!")

	def load(self, filename, sess=None):
		if sess is None:
			raise NotImplementedError('Loading outside a session is not implemented')
			
		with TFSession(sess) as sess:
			super().load(filename, sess=sess)
			self.loss = sess.graph.get_tensor_by_name(self.name + '_loss:0')
			self.accuracy = sess.graph.get_tensor_by_name(self.name + '_accuracy:0')