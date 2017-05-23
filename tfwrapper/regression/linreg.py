import tensorflow as tf

from tfwrapper import TFSession

class LinearRegression():
    def __init__(self, num_independent, num_dependent, num_samples, name='LinearRegression', sess=None):
        self.name = name
        with TFSession(sess) as sess:
            self.X = tf.placeholder(tf.float32, [None, num_independent], name=self.name + '/X_placeholder')
            self.y = tf.placeholder(tf.float32, [None, num_dependent], name=self.name + '/y_placeholder')
            self.lr = tf.placeholder(tf.float32, [], name=self.name + '/learning_rate_placeholder')

            w = tf.Variable(tf.truncated_normal([num_independent, 1]), name=name + '/W')
            b = tf.Variable(tf.truncated_normal([1]), name=name + '/b')
            self.pred = tf.add(tf.matmul(self.X, w), b, name=name + '/pred')

            self.loss = tf.reduce_sum(tf.pow(self.pred-self.y, 2))/(2*num_samples)
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

    def train(self, X, y, *, epochs, sess=None):
        train_len = int(len(X) * 0.8)

        val_X = X[train_len:]
        val_y = y[train_len:]
        X = X[:train_len]
        y = y[:train_len]

        with TFSession(sess) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                _, loss = sess.run([self.optimizer, self.loss], feed_dict={self.lr: 0.000002, self.X: X, self.y: y})
                print('Train loss: %.4f' % loss)
                val_loss = sess.run(self.loss, feed_dict={self.X: val_X, self.y: val_y})
                print('Val loss: %.4f' % val_loss)