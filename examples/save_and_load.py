import os
import numpy as np
import tensorflow as tf

from tfwrapper.models.nets import ShallowCNN
from tfwrapper.datasets import mnist

from utils import curr_path

model_path = os.path.join(curr_path, 'data', 'save_and_load_cnn')


dataset = mnist(size=5000)
dataset = dataset.normalized()
dataset = dataset.balanced()
dataset = dataset.shuffled()
dataset = dataset.translated_labels()
dataset = dataset.onehot_encoded()
train, test = dataset.split(0.8)


cnn = ShallowCNN([28, 28, 1], 10, name='SaveAndLoadExample')
cnn.train(train.X, train.y, epochs=5, keep_prob=0.6)
_, acc = cnn.validate(test.X, test.y)
print('Acc before save: %d%%' % (acc * 100))

if not os.path.isdir(os.path.dirname(model_path)):
    os.mkdir(os.path.dirname(model_path))
cnn.save(model_path)

tf.reset_default_graph()


with tf.Session() as sess:
    loaded_cnn = ShallowCNN([28, 28, 1], 10, sess=sess, name='SaveAndLoadExample')
    loaded_cnn.load(model_path, sess=sess)
    _, acc = loaded_cnn.validate(test.X, test.y, sess=sess)
    print('Acc after load: %d%%' % (acc * 100))