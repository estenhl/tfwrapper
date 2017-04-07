import numpy as np
import tensorflow as tf

from tfwrapper import ImageTransformer
from tfwrapper.nets import CNN
from tfwrapper.datasets import mnist


h = 28
w = 28
c = 1

dataset = mnist(size=1000, verbose=True)
transformer = ImageTransformer(rotation_steps=2, max_rotation_angle=15, blur_steps=2, max_blur_sigma=2.5, hflip=True, vflip=True)
X, y, test_X, test_y, _, _ = dataset.getdata(normalize=True, balance=False, shuffle=True, onehot=True,
                                              split=True, translate_labels=True, transformer=transformer)
X = np.reshape(X, [-1, h, w, c])
num_classes = y.shape[1]

name = 'ExampleCustomCNN'
# TODO: make dependent on list of maxpool factors 'k'
twice_reduce = lambda x: -1 * ((-1 * x) // 4)
layers = layers = [
	CNN.reshape([-1, h, w, c], name=name + '/reshape'),
	CNN.conv2d(filter=[5, 5], input_depth=1, depth=32, name=name + '/conv1'),
	CNN.maxpool2d(k=2, name=name + '/pool1'),
	CNN.conv2d(filter=[5, 5], input_depth=32, depth=64, name=name + '/conv2'),
	CNN.maxpool2d(k=2, name=name + '/pool2'),
	CNN.fullyconnected(input_size=twice_reduce(h)*twice_reduce(w)*64, output_size=512, name=name + '/fc'),
	CNN.out([512, num_classes], num_classes, name=name + '/pred')
]
cnn = CNN([h, w, c], num_classes, layers, name=name)
cnn.learning_rate = 1
cnn.train(X, y, epochs=5, verbose=True)
_, acc = cnn.validate(test_X, test_y)
print('Test accuracy: %d%%' % (acc*100))


