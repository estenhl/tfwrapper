import os
import tensorflow as tf
from tfwrapper.models.nets import ShallowCNN
from tfwrapper.datasets import mnist
from tfwrapper import config

base_path = config.SERVING_MODELS

dataset = mnist(size=10000)
dataset = dataset.normalized()
dataset = dataset.balanced()
dataset = dataset.shuffled()
dataset = dataset.translated_labels()
dataset = dataset.onehot_encoded()
train, test = dataset.split(0.8)

with tf.Session() as sess:
    cnn = ShallowCNN([28, 28, 1], 10, name='TfServingExample', sess=sess)
    cnn.train(train.X, train.y, epochs=20, keep_prob=0.6, sess=sess)
    _, acc = cnn.validate(test.X, test.y, sess=sess)
    print('Acc before save: %d%%' % (acc * 100))

    path = os.path.join(base_path, 'mnist_cnn')
    version = '1'

    cnn.save_serving(os.path.join(path, version), over_write=True, sess=sess)
