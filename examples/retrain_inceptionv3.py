import os
import tensorflow as tf

from tfwrapper import Dataset
from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.nets.pretrained import InceptionV3
from tfwrapper.datasets import checkboxes

curr_path = os.path.dirname(os.path.realpath(__file__))
inception_path = os.path.join(curr_path, 'data', 'inception_v3_features.csv')
model_path = os.path.join(curr_path, 'data', 'inception')
if not os.path.isdir(os.path.join(curr_path, 'data')):
	os.mkdir(os.path.join(curr_path, 'data'))

_, data_folder = checkboxes()
inception = InceptionV3()
features = inception.extract_features_from_datastructure(data_folder, feature_file=inception_path)

dataset = Dataset(features=features)
X, y, test_X, test_y, labels =dataset.getdata(normalize=True, balance=True, translate_labels=True, shuffle=True, onehot=True, split=True)

graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		nn = SingleLayerNeuralNet([X.shape[1]], 2, 1024, sess=sess, graph=graph, name='InceptionV3Test')
		nn.train(X, y, epochs=10, sess=sess, verbose=True)
		nn.save(model_path, sess=sess)

nn = SingleLayerNeuralNet([X.shape[1]], 2, 1024, name='InceptionV3Test')
graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		nn.load(model_path, sess=sess)
		_, acc = nn.validate(test_X, test_y, sess=sess)
		print('Inception Test accuracy: %d %%' % (acc*100))