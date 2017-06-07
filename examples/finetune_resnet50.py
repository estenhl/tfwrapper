import numpy as np
import tensorflow as tf

from tfwrapper.models.nets import ResNet50
from tfwrapper.layers import randomized_preprocessing
from tfwrapper.layers import vgg_preprocessing
from tfwrapper.datasets import cifar10
from tfwrapper.hyperparameters import adjust_at_epochs

train = cifar10()
train = train.balance()
train = train.shuffle()
train = train.translate_labels()
train = train.onehot()

test = cifar10(test=True)
test = test.translate_labels()
test = test.onehot()

with tf.Session() as sess:
    cnn = ResNet50.from_h5(X_shape=[32, 32, 3], classes=10, name='FinetunedResNet50', sess=sess)
    cnn.learning_rate = adjust_at_epochs([20, 50, 100], [0.00005, 0.00003, 0.00001, 0.000005])
    cnn.batch_size = 256
    cnn.train(train.X, train.y, validate=0.05, epochs=500, sess=sess)
    _, acc = cnn.validate(test.X, test.y, sess=sess)
    print('Test accuracy: %d%%' % (acc*100))