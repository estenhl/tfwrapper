import os
import tensorflow as tf

from tfwrapper import ImageDataset
from tfwrapper import FeatureLoader
from tfwrapper import ImagePreprocessor
from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.nets.pretrained import InceptionV4
from tfwrapper.datasets import cats_and_dogs

from utils import curr_path

dataset = cats_and_dogs()
dataset = dataset.balance(max=50)
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

features_file = os.path.join(curr_path, 'data', 'catsdogs_inceptionv4.csv')

with tf.Session() as sess:
    inception = InceptionV4()

    train_prep = ImagePreprocessor()
    train_prep.resize_to = (299, 299)
    train_prep.flip_lr = True
    train_loader = FeatureLoader(inception, cache=features_file, preprocessor=train_prep, sess=sess)
    train.loader = train_loader
    train.loader.sess = sess
    X, y = train.X, train.y

    test_prep = ImagePreprocessor()
    test_prep.resize_to = (299, 299)
    test_loader = FeatureLoader(inception, cache=features_file, preprocessor=test_prep, sess=sess)
    test.loader = test_loader
    test.loader.sess = sess
    test_X, test_y = test.X, test.y

with tf.Session() as sess:
    nn = SingleLayerNeuralNet([1536], 2, 1024, sess=sess, name='InceptionV4Test')
    nn.train(X, y, epochs=10, sess=sess, verbose=True)
    _, acc = nn.validate(test_X, test_y, sess=sess)
    print('Acc: %d%%' % (acc * 100))

