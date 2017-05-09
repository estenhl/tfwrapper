import numpy as np
import tensorflow as tf

from tfwrapper.nets import SqueezeNet
from tfwrapper.datasets import cifar100

dataset = cifar100()
dataset = dataset.normalize()
dataset = dataset.balance()
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

cnn = SqueezeNet([32, 32, 3], 10, name='ExampleSqueezeNet')
cnn.learning_rate = 0.1
cnn.train(train.X, train.y, epochs=5, verbose=True)
_, acc = cnn.validate(test.X, test.y)
preds = cnn.predict(test.X)
print('Test accuracy: %d%%' % (acc*100))
for pred in preds:
    print(pred)
