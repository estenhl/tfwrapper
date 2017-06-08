import numpy as np
import tensorflow as tf

from tfwrapper import logger
from tfwrapper.models.nets import ShallowCNN
from tfwrapper.datasets import cifar10

dataset = cifar10(size=100)
dataset = dataset.normalize()
dataset = dataset.balance()
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

cnn = ShallowCNN([32, 32, 3], 10, name='ExampleShallowCNN')
cnn.train(train.X, train.y, epochs=5)
_, acc = cnn.validate(test.X, test.y)
print('Test accuracy: %d%%' % (acc*100))


