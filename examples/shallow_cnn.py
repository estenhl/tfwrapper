import numpy as np
import tensorflow as tf

from tfwrapper import logger
from tfwrapper.models.nets import ShallowCNN
from tfwrapper.datasets import cifar10

dataset = cifar10()
dataset = dataset.normalized()
dataset = dataset.balanced()
dataset = dataset.shuffled()
dataset = dataset.translated_labels()
dataset = dataset.onehot_encoded()
train, test = dataset.split(0.8)

cnn = ShallowCNN([32, 32, 3], 10, name='ExampleShallowCNN')
cnn.train(train.X, train.y, keep_prob=0.8, epochs=5)
_, acc = cnn.validate(test.X, test.y)
print('Test accuracy: %d%%' % (acc*100))


