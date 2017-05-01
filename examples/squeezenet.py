import numpy as np
import tensorflow as tf

from tfwrapper.nets import SqueezeNet
from tfwrapper.datasets import mnist

dataset = mnist(size=10000,  verbose=True, imagesize=[224, 224])
dataset = dataset.normalize()
dataset = dataset.balance()
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

cnn = SqueezeNet([224, 224, 1], 10, name='ExampleSqueezeNet')
cnn.train(train.X, train.y, epochs=1, verbose=True)
_, acc = cnn.validate(test.X, test.y)
print('Test accuracy: %d%%' % (acc*100))

