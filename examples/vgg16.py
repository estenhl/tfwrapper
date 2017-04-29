import cv2
import numpy as np
import tensorflow as tf

from tfwrapper import ImageLoader
from tfwrapper import ImagePreprocessor
from tfwrapper.nets import VGG16
from tfwrapper.datasets import mnist


X_shape = [224, 224, 3]
dataset = mnist(size=1000, verbose=True)
dataset = dataset.balance()
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

X = np.zeros([len(train.X), 224, 224, 1])
for i in range(len(X)):
    X[i] = np.resize(cv2.resize(train.X[i], (224, 224)), (224, 224, 1))

with tf.Session() as sess:
	cnn = VGG16([224, 224, 1], classes=10, sess=sess)
	cnn.train(X, train.y, epochs=10, sess=sess, verbose=True)