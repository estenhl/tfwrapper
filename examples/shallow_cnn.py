import numpy as np
import tensorflow as tf

from tfwrapper.nets import ShallowCNN
from tfwrapper.datasets import mnist

dataset = mnist(size=10000, verbose=True)
X, y, test_X, test_y, _ = dataset.getdata(normalize=True, balance=True, shuffle=True, onehot=True, split=True)
X = np.reshape(X, [-1, 28, 28, 1])

cnn = ShallowCNN([28, 28, 1], 10, name='ExampleShallowCNN')
cnn.train(X, y, epochs=5, verbose=True)
_, acc = cnn.validate(test_X, test_y)
print('Test accuracy: %d%%' % (acc*100))


