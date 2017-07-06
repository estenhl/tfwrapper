import numpy as np
import tensorflow as tf

from tfwrapper.models.nets import SqueezeNet
from tfwrapper.datasets import cifar10

dataset = cifar10()
dataset = dataset.normalized()
dataset = dataset.balanced()
dataset = dataset.shuffled()
dataset = dataset.translated_labels()
dataset = dataset.onehot_encoded()
train, test = dataset.split(0.8)
generator = train.batch_generator(128, normalize=True, shuffle=True)

with tf.Session() as sess:
    cnn = SqueezeNet([32, 32, 3], 10, name='ExampleSqueezeNet', sess=sess)
    cnn.learning_rate = 0.00005
    cnn.train(generator=generator, epochs=100, sess=sess)
    _, acc = cnn.validate(test.X, test.y, sess=sess)
    print('Test accuracy: %d%%' % (acc*100))
    preds = cnn.predict(test.X, sess=sess)

