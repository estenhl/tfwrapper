import tensorflow as tf

from tfwrapper.datasets import boston
from tfwrapper.models.regression import LinearRegression
from tfwrapper.layers.loss import mse

dataset = boston()
dataset = dataset.normalized()
train, test = dataset.split(0.8)

with tf.Session() as sess:
    model = LinearRegression(13, 1, sess=sess)
    model.learning_rate = 0.05
    model.loss = mse
    model.fit(train.X, train.y, validate=True, epochs=1000, sess=sess)

    loss = model.validate(test.X, test.y, sess=sess)
    print('Loss: %.2f' % loss)

    preds = model.predict(test.X, sess=sess)
    print('Preds: %s' % str(preds[:5]))

    