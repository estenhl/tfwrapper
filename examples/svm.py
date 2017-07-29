from tfwrapper.datasets import iris
from tfwrapper.models.regression import MulticlassSVM

dataset = iris()
dataset = dataset.shuffled()
dataset = dataset.onehot_encoded()
train, test = dataset.split(0.8)

import tensorflow as tf
with tf.Session() as sess:
    model = MulticlassSVM(dataset.X.shape[1], 3, name='ExampleSVM', sess=sess)
    model.learning_rate = 0.01
    model.fit(train.X, train.y, epochs=1000, sess=sess)

    loss = model.validate(test.X, test.y, sess=sess)
    print('Loss: %.2f' % loss)

