import math
import numpy as np
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.layers.loss import multiclass_hinge

class MulticlassSVM():
    batch_size = 512
    learning_rate = 0.01

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, val):
        self._loss = val(self.y, self.pred, name=self.name + '/loss')

    def __init__(self, num_variables, num_classes, name='LinearRegression', sess=None):
        self.name = name

        with TFSession(sess) as sess:
            self.X = tf.placeholder(tf.float32, [None, num_variables], name=name + '/X_placeholder')
            self.y = tf.placeholder(tf.float32, [None, num_classes], name=name + '/y_placeholder')
            self.lr = tf.placeholder(tf.float32, [], name=name + '/learning_rate_placeholder')

            W = tf.Variable(tf.ones([num_variables, num_classes]), name=name + '/W')
            b = tf.Variable(tf.zeros([num_classes]), name=name + '/b')

            self.pred = tf.add(tf.matmul(self.X, W, name=name + '/matmul'), b, name=name + '/pred')
            self._loss = multiclass_hinge(self.y, self.pred)
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr, name=name + '/optimizer').minimize(self.loss)

            self.graph = sess.graph

    def fit(self, X, y, *, validate=False, epochs, sess=None):
        #with tf.Session(graph=self.graph) as sess:
            #sess.run(tf.global_variables_initializer())
            #multiclass_hinge(self.y, self.pred, feed_dict={self.X: X, self.y: y})
        #exit()
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)

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
                    _, train_loss = sess.run([self.optimizer, self.loss], feed_dict={self.X: X_batches[i], self.y: y_batches[i], self.lr: self.learning_rate})
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
            loss = sess.run(self.loss, feed_dict={self.X: X, self.y: y})

        return loss