import os
import numpy as np
import tensorflow as tf

from tfwrapper import FeatureLoader
from tfwrapper import ImagePreprocessor
from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.nets.pretrained import PretrainedVGG16
from tfwrapper.datasets import imagenet
from tfwrapper.datasets import cats_and_dogs

from utils import curr_path

dataset = cats_and_dogs(size=500)
dataset = dataset.balance(max=150)
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

datafolder = os.path.join(curr_path, 'data')
if not os.path.isdir(datafolder):
    os.mkdir(datafolder)
features_file = os.path.join(datafolder, 'catsdogs_vgg16.csv')

with tf.Session() as sess:
    vgg = PretrainedVGG16([224, 224, 3], name='PretrainedVGG16', sess=sess)

    train_prep = ImagePreprocessor()
    train_prep.resize_to = (224, 224)
    train_prep.flip_lr = True
    train_loader = FeatureLoader(vgg, cache=features_file, preprocessor=train_prep, sess=sess)
    train.loader = train_loader
    train.loader.sess = sess
    X, y = train.X, train.y

    test_prep = ImagePreprocessor()
    test_prep.resize_to = (224, 224)
    test_loader = FeatureLoader(vgg, cache=features_file, preprocessor=test_prep, sess=sess)
    test.loader = test_loader
    test.loader.sess = sess
    test_X, test_y = test.X, test.y

with tf.Session() as sess:
    nn = SingleLayerNeuralNet([4096], 2, 1024, sess=sess, name='InceptionV3Test')
    nn.train(X, y, epochs=10, sess=sess)
    _, acc = nn.validate(test_X, test_y, sess=sess)
    print('Acc: %d%%' % (acc * 100))
