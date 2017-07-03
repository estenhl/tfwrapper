import tensorflow as tf

from tfwrapper.datasets import boston
from tfwrapper.models.regression import LinearRegression
from tfwrapper.layers.loss import mse
from tfwrapper.visualization import plot_regression_predictions

dataset = boston()
dataset = dataset.normalized(columnwise=True)
dataset = dataset.shuffle()
train, test = dataset.split(0.8)

with tf.Session() as sess:
    model = LinearRegression(13, 1, sess=sess)
    model.learning_rate = 0.01
    model.fit(train.X, train.y, validate=True, epochs=1000, sess=sess)

    loss = model.validate(test.X, test.y, sess=sess)
    print('Loss: %.2f' % loss)

    preds = model.predict(test.X, sess=sess)
    plot_regression_predictions(test.y, preds, colour='green', title='Linear regression example', figsize=(10, 10))

    