import tensorflow as tf

from .neural_net import NeuralNet
from tfwrapper import TFSession
from tfwrapper.layers import lstm_layer

class LSTM(NeuralNet):
    def __init__(self, X_shape=None, y_size=None, hidden=None, sess=None, name='LSTM'):
        super().__init__(name=name)
        if X_shape is not None and y_size is not None:
            with TFSession(sess) as sess:
                self.fill_from_shape(sess, X_shape, y_size, hidden)
                self.post_init()

    @classmethod
    def from_shape(cls, X_shape, y_size, hidden, sess=None, name='LSTM'):
        model = cls(X_shape=None, y_size=None, name=name)
        model.fill_from_shape(sess=sess, X_shape=X_shape, y_size=y_size, hidden=hidden)
        model.post_init()
        return model

    def fill_from_shape(self, sess, X_shape, y_size, hidden):
        layers = [lstm_layer(hidden, y_size)]
        super().fill_from_shape(sess, X_shape, y_size, layers)

    def optimizer_function(self):
        cross_entropy = -tf.reduce_sum(self.y * tf.log(tf.clip_by_value(self.pred,1e-10,1.0)))

        return tf.train.AdamOptimizer().minimize(cross_entropy)

    def accuracy_function(self):
        return lambda x: 0.0
