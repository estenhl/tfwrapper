import cv2
import numpy as np
import tensorflow as tf

from tfwrapper import ImageLoader
from tfwrapper import ImagePreprocessor
from tfwrapper.nets import VGG16
from tfwrapper.datasets import mnist


dataset = mnist(size=10000, verbose=True)
dataset = dataset.balance()
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

X = np.zeros([len(train.X),32, 32, 1])
for i in range(len(X)):
    X[i] = np.resize(cv2.resize(train.X[i], (32, 32)), (32, 32, 1))

test_X = np.zeros([len(test.X), 32, 32, 1])
for i in range(len(test_X)):
    test_X[i] = np.resize(cv2.resize(test.X[i], (32, 32)), (32, 32, 1))

with tf.Session() as sess:
    cnn = VGG16([32, 32, 1], classes=10, sess=sess)
    cnn.learning_rate = 0.0001
    cnn.train(X, train.y, epochs=50, sess=sess, verbose=True)
    _, acc = cnn.validate(test_X, test.y)
    print('Test accuracy: %d%%' % (acc*100))