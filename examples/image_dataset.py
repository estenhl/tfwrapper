import tensorflow as tf

from tfwrapper.datasets.image_dataset import ImageDataset
from tfwrapper.datasets.image_augment import ImagePreprocess
from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.nets.pretrained.inception_v4 import Inception_v4, FEATURE_LAYER
from tfwrapper.datasets.generator import CachedFeatureGenerator, ImageGenerator

import os

dataset_path = "path/to/dataset"
feature_cache = "path/to/feature_cache.csv"
inception_file = "path/to/inception_v4" #can be default value in model
model_path = "path/to/model"

dataset = ImageDataset(dataset_path)
# train = train.balance_dataset(max_value=400)
dataset.shuffle()
dataset.one_hot_encode()

dataset = dataset.balance_dataset(max_value=1000) #should also be possible to choose max_value == min of smallest class


train, test = dataset.split(shape=[0.85, 0.15])

img_size = (299, 299)

train_preprocess = ImagePreprocess()
train_preprocess.resize(img_size=img_size)
train_preprocess.append_flip_lr()

test_preprocess = ImagePreprocess()
test_preprocess.resize(img_size=img_size)

inc_v4 = Inception_v4(inception_file)
generator = CachedFeatureGenerator(feature_cache, inc_v4, FEATURE_LAYER) #Feature layer could be defaulted in model

train_X, train_Y, names = generator.load(train, train_preprocess)

test_X, test_Y, names = generator.load(test, test_preprocess)


graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		nn = SingleLayerNeuralNet([train_X.shape[1]], 2, 1024, sess=sess, graph=graph, name='InceptionV3Test')
		nn.train(train_X, train_Y, epochs=10, sess=sess, verbose=True)
		nn.save(model_path, sess=sess)

nn = SingleLayerNeuralNet([test_X.shape[1]], 2, 1024, name='InceptionV3Test')
graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		nn.load(model_path, sess=sess)
		_, acc = nn.validate(test_X, test_Y, sess=sess)
		print('Inception Test accuracy: %d %%' % (acc*100))

#Train model