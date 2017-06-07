import cv2
import numpy as np
import tensorflow as tf

from tfwrapper.nets import VGG16
from tfwrapper.datasets import cifar100


dataset = cifar100()
dataset = dataset.balance()
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)


with tf.Session() as sess:
    cnn = VGG16([32, 32, 3], classes=100, sess=sess)
    cnn.learning_rate = 0.01
    cnn.batch_size = 512
    cnn.train(train.X, train.y, epochs=50, sess=sess)
    _, acc = cnn.validate(test.X, test.y)
    print('Test accuracy: %d%%' % (acc*100))
