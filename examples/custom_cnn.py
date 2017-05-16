import numpy as np
import tensorflow as tf

from tfwrapper.nets import CNN
from tfwrapper.layers import reshape, conv2d, maxpool2d, fullyconnected, out
from tfwrapper.datasets import mnist

h = 28
w = 28
c = 1

dataset = mnist(size=10000)
dataset = dataset.normalize()
dataset = dataset.balance()
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

X = train.X
y = train.y

num_classes = y.shape[1]

name = 'ExampleCustomCNN'
# TODO: make dependent on list of maxpool factors 'k'
twice_reduce = lambda x: -1 * ((-1 * x) // 4)
layers = [
	reshape([-1, h, w, c], name=name + '/reshape'),
	conv2d(filter=[5, 5], depth=32, name=name + '/conv1'),
	maxpool2d(k=2, name=name + '/pool1'),
	conv2d(filter=[5, 5], depth=64, name=name + '/conv2'),
	maxpool2d(k=2, name=name + '/pool2'),
	fullyconnected(inputs=twice_reduce(h)*twice_reduce(w)*64, outputs=512, name=name + '/fc'),
	out(inputs=512, outputs=num_classes, name=name + '/pred')
]
cnn = CNN([h, w, c], num_classes, layers, name=name)
cnn.learning_rate = 0.001
cnn.train(X, y, epochs=5, validate=False)
_, acc = cnn.validate(test.X, test.y)
print('Test accuracy: %d%%' % (acc*100))



