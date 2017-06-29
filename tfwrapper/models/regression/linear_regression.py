import math
import numpy as np
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.layers.loss import mse

class LinearRegression():
    batch_size = 512
    learning_rate = 0.01

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, val):
        self._loss = val(self.y, self.pred, self.num_samples, name=self.name + '/loss')

    def __init__(self, num_independent, num_dependent, name='LinearRegression', sess=None):
        self.name = name

        with TFSession(sess) as sess:
            self.X = tf.placeholder(tf.float32, [None, num_independent], name=name + '/X_placeholder')
            self.y = tf.placeholder(tf.float32, [None, num_dependent], name=name + '/y_placeholder')
            self.num_samples = tf.placeholder(tf.float32, [], name=name + '/num_samples_placeholder')
            self.lr = tf.placeholder(tf.float32, [], name=name + '/learning_rate_placeholder')

            W = tf.Variable(tf.ones([num_independent, num_dependent]), name=name + '/W')
            b = tf.Variable(tf.zeros([num_dependent]), name=name + '/b')

            self.pred = tf.add(tf.matmul(self.X, W, name=name + '/matmul'), b, name=name + '/pred')
            self._loss = mse(self.y, self.pred, self.num_samples, name=name + '/loss')
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr, name=name + '/optimizer').minimize(self.loss)

            self.graph = sess.graph

    def fit(self, X, y, *, validate=False, epochs, sess=None):
        num_batches = math.ceil(len(X) / self.batch_size)
        num_samples = len(X)

        if validate:
            if type(validate) is bool:
                validate = 0.8
            train_len = int(len(X) * validate)
            val_X = X[train_len:]
            val_y = y[train_len:]
            X = X[:train_len]
            y = y[:train_len]

        X_batches = np.array_split(X, num_batches)
        y_batches = np.array_split(y, num_batches)

        with TFSession(sess, self.graph, init=True) as sess:
            for epoch in range(epochs):
                for i in range(num_batches):
                    _, train_loss = sess.run([self.optimizer, self.loss], feed_dict={self.X: X_batches[i], self.y: y_batches[i], self.num_samples: num_samples, self.lr: self.learning_rate})
                summary = 'Epoch %d: train loss %f' % (epoch, train_loss)
                
                if validate:
                    val_loss = self.validate(val_X, val_y, sess=sess)
                    summary += ', val loss %f' % val_loss
                
                print(summary)

    def predict(self, X, sess=None):
        with TFSession(sess, self.graph) as sess:
            preds = sess.run(self.pred, feed_dict={self.X: X})

        return preds

    def validate(self, X, y, sess=None):
        num_samples = len(X)

        with TFSession(sess, self.graph) as sess:
            loss = sess.run(self.loss, feed_dict={self.X: X, self.y: y, self.num_samples: num_samples})

        return loss