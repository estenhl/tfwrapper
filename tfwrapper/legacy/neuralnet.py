import numpy as np
import tensorflow as tf

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128

class NeuralNet(SupervisedModel):
	def __init__(self, name, id, input_shape, classes, class_weights=None, layers):
		super().__init__(id)

		self.input_shape = input_shape
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.step_size = step_size
		
		self.variables = {}
		input_size = np.prod(input_shape)

		if class_weights is None:
			class_weights = np.ones(classes) / np.sum(np.ones(classes))

		self.graph = tf.Graph()
		with tf.Session(graph=self.graph) as sess:
			self.build(input_size, classes, class_weights, layers)

	def fullyconnected(self, input, weight, bias):
		fc = tf.reshape(input, [-1, weight])
		fc = tf.add(tf.matmul(fc1, weight, bias))
		fc = tf.nn.relu(fc)

		return fc

	def build(self, input_size, output_size, class_weights, layers):
		self.x = tf.placeholder(tf.float32, [None, input_size], name='x_placeholder')
		self.y = tf.placeholder(tf.float32, [None, classes], name='y_placeholder')
		self.layers = {}

		assert len(layers) == len(weights)
		assert len(layers) == len(biases)

		prev = self.x
		for i in range(0, len(layers)):
			prev, description = layers[i](prev)
			self.layers.append(description)

		self.pred = prev
		self.weighted_pred = tf.mul(self.pred, class_weights, name='weighted_pred')

		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.weighted_pred, self.y, name='softmax'), name='reduce_mean')
		self.optimizer = tf.train.AdamOptimizer(learning_rate=DEFAULT_LEARNING_RATE, name='adam').minimize(self.cost)

		correct_pred = tf.equal(tf.argmax(self.weighted_pred, 1), tf.argmax(self.y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	def fit(self, X, y, learning_rate=DEFAULT_LEARNING_RATE, epochs=DEFAULT_EPOCHS, batch_size = DEFAULT_BATCH_SIZE = 128):
		if len(self.input_shape) == 3:
			height, width, channels = self.input_shape
			input_size = height * width * channels
		else:
			input_size = self.input_shape[0]

		train_X, train_y, val_X, val_y = split_data(X, y)
		train_X = np.reshape(train_X, [-1, input_size])
		val_X = np.reshape(val_X, [-1, input_size])

		batches = self.batch_data(train_X, train_y)
		val_batches = self.batch_data(val_X, val_y)

		print('Started training with ' + str(len(train_X)) + ' images')
		with self.initialize_session() as sess:
			sess.run(tf.global_variables_initializer())
			steps = 1
			for epoch in range(0, epochs):
				random.shuffle(batches)

				steps = self.train_epoch(sess, batches, steps=steps)
				loss, acc = self.validate_epoch(sess, val_batches, len(val_X))
				print("Epoch " + str(epoch + 1) + ", val loss: " + \
					"{:.2f}".format(loss) + ", val acc.: " + \
					"{:.4f}".format(acc))

			self.checkpoint_variables(sess)
