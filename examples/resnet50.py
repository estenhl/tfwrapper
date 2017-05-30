import numpy as np
import tensorflow as tf

from tfwrapper.nets import ResNet50
from tfwrapper.datasets import cifar10
from tfwrapper.hyperparameters import adjust_after_epoch

train = cifar10()
train = train.balance()
train = train.shuffle()
train = train.translate_labels()
train = train.onehot()

test = cifar10(test=True)
test = test.translate_labels()
test = test.onehot()

with tf.Session() as sess:
    cnn = ResNet50([32, 32, 3], 10, name='ExampleResNet50', sess=sess)
    cnn.learning_rate = adjust_after_epoch(150, before=0.000005, after=0.000001)
    cnn.batch_size = 256
    cnn.train(train.X, train.y, validate=0.05, epochs=300, sess=sess)
    _, acc = cnn.validate(test.X, test.y, sess=sess)
    print('Test accuracy: %d%%' % (acc*100))
    preds = cnn.predict(test.X, sess=sess)

