import numpy as np
import tensorflow as tf

def generate_dataset(size=1000):
	X = []
	y = []

	for i in range(0, size/2):
		X.append(np.ones((48, 48)))
		y.append(1)

	for i in range(0, size/2):
		X.append(np.zeros((48, 48)))
		y.append(0)

	return Dataset(X=np.asarray(X), y=np.asarray(y))

dataset = generate_dataset()
X, y, test_X, test_y = dataset.getdata(shuffle=True, onehot=True, split=True)

graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		cnn = DeepCNN([48, 48], 2, sess=sess, graph=graph, name='ExampleCNN')
		cnn.train(X, y, epochs=100, verbose=True)
