import tensorflow as tf
from .cnn import CNN

class DeepCNN(CNN):
	def __init__(self, id, input_shape, classes, class_weights=None):
		super().__init__(id, input_shape, classes, class_weights=class_weights)

	def weights(self):
		height, width, channels = self.input_shape
		return {
			'wc1': tf.Variable(tf.random_normal([5, 5, channels, 32]), name='wc1'),
			'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name='wc2'),
			'wc3': tf.Variable(tf.random_normal([5, 5, 64, 64]), name='wc3'),
			'wc4': tf.Variable(tf.random_normal([5, 5, 64, 128]), name='wc4'),
			'wc5': tf.Variable(tf.random_normal([5, 5, 128, 128]), name='wc5'),
			'wc6': tf.Variable(tf.random_normal([5, 5, 128, 256]), name='wc6'),
			'wc7': tf.Variable(tf.random_normal([3, 3, 256, 256]), name='wc7'),
			'wc8': tf.Variable(tf.random_normal([3, 3, 256, 512]), name='wc8'),
			'wd1': tf.Variable(tf.random_normal([512, 1024]), name='wd1'),
			'wd2': tf.Variable(tf.random_normal([1024, 512]), name='wd2'),
			'out': tf.Variable(tf.random_normal([512, self.classes]), name='out_weight')
		}

	def biases(self):
		return {
			'bc1': tf.Variable(tf.random_normal([32]), name='bc1'),
			'bc2': tf.Variable(tf.random_normal([64]), name='bc2'),
			'bc3': tf.Variable(tf.random_normal([64]), name='bc3'),
			'bc4': tf.Variable(tf.random_normal([128]), name='bc4'),
			'bc5': tf.Variable(tf.random_normal([128]), name='bc5'),
			'bc6': tf.Variable(tf.random_normal([256]), name='bc6'),
			'bc7': tf.Variable(tf.random_normal([256]), name='bc7'),
			'bc8': tf.Variable(tf.random_normal([512]), name='bc8'),
			'bd1': tf.Variable(tf.random_normal([1024]), name='bd1'),
			'bd2': tf.Variable(tf.random_normal([512]), name='bd2'),
			'out': tf.Variable(tf.random_normal([self.classes]), name='out_bias')
		}

	def net(self, x, input_shape, weights, biases):
		height, width, channels = input_shape
		x = tf.reshape(x, shape=[-1, height, width, channels])
		layers = []

		# Conv1
		conv1 = self.conv2d(x, weights['wc1'], biases['bc1'], name='conv1')
		depth = weights['wc1'].get_shape().as_list()[3]
		size = str(input_shape[0]) + 'x' + str(input_shape[1]) + 'x' + str(depth)
		layers.append({'name': 'conv1', 'size': size})

		# Pool1
		k1 = 2
		pool1 = self.maxpool2d(conv1, k=k1, name='pool1')
		size = str(int(input_shape[0]/k1)) + 'x' +  str(int(input_shape[1]/k1)) + 'x' + str(depth)
		layers.append({'name': 'pool1', 'size': size})

		# Conv2
		conv2 = self.conv2d(pool1, weights['wc2'], biases['bc2'], name='conv2')
		depth = weights['wc2'].get_shape().as_list()[3]
		size = str(int(input_shape[0]/k1)) + 'x' +  str(int(input_shape[1]/k1)) + 'x' + str(depth)
		layers.append({'name': 'conv2', 'size': size})

		# Pool2
		k2 = 2
		pool2 = self.maxpool2d(conv2, k=k2, name='pool2')
		size = str(int(input_shape[0]/(k1*k2))) + 'x' +  str(int(input_shape[1]/(k1*k2))) + 'x' + str(depth)
		layers.append({'name': 'pool2', 'size': size})

		# Conv3
		conv3 = self.conv2d(pool2, weights['wc3'], biases['bc3'], name='conv3')
		depth = weights['wc3'].get_shape().as_list()[3]
		size = str(int(input_shape[0]/(k1*k2))) + 'x' +  str(int(input_shape[1]/(k1*k2))) + 'x' + str(depth)
		layers.append({'name': 'conv3', 'size': size})

		# Conv4
		conv4 = self.conv2d(pool2, weights['wc4'], biases['bc4'], name='conv4')
		depth = weights['wc4'].get_shape().as_list()[3]
		size = str(int(input_shape[0]/(k1*k2))) + 'x' +  str(int(input_shape[1]/(k1*k2))) + 'x' + str(depth)
		layers.append({'name': 'conv4', 'size': size})

		# Pool3
		k3 = 2
		pool3 = self.maxpool2d(conv4, k=k3, name='pool3')
		size = str(int(input_shape[0]/(k1*k2*k3))) + 'x' +  str(int(input_shape[1]/(k1*k2*k3))) + 'x' + str(depth)
		layers.append({'name': 'pool3', 'size': size})

		# Conv5
		conv5 = self.conv2d(pool3, weights['wc5'], biases['bc4'], name='conv5')
		depth = weights['wc5'].get_shape().as_list()[3]
		size = str(int(input_shape[0]/(k1*k2*k3))) + 'x' +  str(int(input_shape[1]/(k1*k2*k3))) + 'x' + str(depth)
		layers.append({'name': 'conv5', 'size': size})

		# Conv6
		conv6 = self.conv2d(conv5, weights['wc6'], biases['bc6'], name='conv6')
		depth = weights['wc6'].get_shape().as_list()[3]
		size = str(int(input_shape[0]/(k1*k2*k3))) + 'x' +  str(int(input_shape[1]/(k1*k2*k3))) + 'x' + str(depth)
		layers.append({'name': 'conv6', 'size': size})

		# Pool 4
		#k4 = 2
		#pool4 = self.maxpool2d(conv6, k=k4, name='pool4')
		#size = str(int(input_shape[0]/(k1*k2*k3*k4))) + 'x' +  str(int(input_shape[1]/(k1*k2*k3*k4))) + 'x' + str(depth)
		#layers.append({'name': 'pool4', 'size': size})

		# Conv7
		conv7 = self.conv2d(conv6, weights['wc7'], biases['bc7'], name='conv7')
		depth = weights['wc7'].get_shape().as_list()[3]
		size = str(int(input_shape[0]/(k1*k2*k3))) + 'x' +  str(int(input_shape[1]/(k1*k2*k3))) + 'x' + str(depth)
		layers.append({'name': 'conv7', 'size': size})

		# Conv8
		conv8 = self.conv2d(conv7, weights['wc8'], biases['bc8'], name='conv8')
		depth = weights['wc8'].get_shape().as_list()[3]
		size = str(int(input_shape[0]/(k1*k2*k3))) + 'x' +  str(int(input_shape[1]/(k1*k2*k3))) + 'x' + str(depth)
		layers.append({'name': 'conv8', 'size': size})

		# Flatten
		k5 = input_shape[0]/(k1*k2*k3)
		flatten = self.maxpool2d(conv8, k=k5, name='flatten')
		size = '1x1x' + str(depth)
		layers.append({'name': 'flatten', 'size': size})

		# Fully connected 1
		fc1 = tf.reshape(flatten, [-1, weights['wd1'].get_shape().as_list()[0]])
		fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
		fc1 = tf.nn.relu(fc1)
		size = str(weights['wd1'].get_shape().as_list()[1])
		layers.append({'name': 'fc1', 'size': size})

		# Fully connected 2
		fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
		fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
		fc2 = tf.nn.relu(fc2)
		fc2 = tf.nn.dropout(fc2, 1, name='dropout')
		size = str(weights['wd2'].get_shape().as_list()[1])
		layers.append({'name': 'fc2', 'size': size})

		# Output
		out = tf.add(tf.matmul(fc2, weights['out']), biases['out'], name='out')
		size = str(weights['out'].get_shape().as_list()[1])
		layers.append({'name': 'out', 'size': size})
		
		return out, layers



