import numpy as np
import tensorflow as tf

from tfwrapper.nets import SqueezeNet
from tfwrapper.datasets import cifar10

dataset = cifar10()
dataset = dataset.normalize()
dataset = dataset.balance()
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

with tf.Session() as sess:
    cnn = SqueezeNet([32, 32, 3], 10, name='ExampleSqueezeNet', sess=sess)
    cnn.learning_rate = 0.00005
    cnn.train(train.X, train.y, epochs=100, sess=sess)
    _, acc = cnn.validate(test.X, test.y, sess=sess)
    print('Test accuracy: %d%%' % (acc*100))
    preds = cnn.predict(test.X, sess=sess)

