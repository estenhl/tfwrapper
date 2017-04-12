import numpy as np
import tensorflow as tf

from tfwrapper.nets import ShallowCNN
from tfwrapper.datasets import mnist

dataset = mnist(size=5000, verbose=True)
dataset = dataset.normalize()
dataset = dataset.balance()
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

X = train.X
y = train.y
X = np.reshape(X, [-1, 28, 28, 1])

cnn = ShallowCNN([28, 28, 1], 10, name='ExampleShallowCNN')
cnn.train(X, y, epochs=10, verbose=True)
_, acc = cnn.validate(test.X, test.y)
print('Test accuracy: %d%%' % (acc*100))


