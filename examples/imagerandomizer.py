import numpy as np

from tfwrapper import twimage
from tfwrapper.nets import SqueezeNet
from tfwrapper.nets import ImageRandomizer
from tfwrapper.datasets import cifar10

train = cifar10()
train = train.translate_labels()
train = train.onehot()

test = cifar10(test=True)
test = test.translate_labels()
test = test.onehot()

randomizer = ImageRandomizer([32, 32, 3], flip_lr=True, brightness_delta=1, hue_delta=0.2, contrast_range=[0.7, 1], saturation_range=[0.7, 1.3])
cnn = randomizer + SqueezeNet([32, 32, 3], 10, name='SqueezeNet')
cnn.learning_rate = 0.00005
cnn.train(train.X, train.y, validate=0.05, keep_prob=0.8, epochs=100)

