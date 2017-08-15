import json
import numpy as np
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.layers import fullyconnected
from tfwrapper.layers import out

from .neural_net import NeuralNet

class SingleLayerNeuralNet(NeuralNet):
    def __init__(self, X_shape, y_size, hidden, sess=None, name='SingleLayerNeuralNet'):
        self.hidden = hidden

        if type(y_size) is list:
            y_size = int(np.prod(np.asarray(y_size)))

        X_size = np.prod(X_shape)
        layers = [
            fullyconnected(inputs=X_size, outputs=hidden, name=name + '/hidden'),
            fullyconnected(inputs=hidden, outputs=y_size, activation=None, name=name + '/pred')
        ]
        with TFSession(sess) as sess:
            super().__init__(X_shape, y_size, layers, sess=sess, name=name)
            self._graph = sess.graph

    @classmethod
    def from_tw_data(cls, data, weights_filename, sess=None, **kwargs):
        kwargs['hidden'] = data['hidden']

        with TFSession(sess) as sess:
            return super().from_tw_data(data, weights_filename, sess=sess, **kwargs)

    def save(self, filename, sess=None, **kwargs):
        kwargs['hidden'] = self.hidden

        with TFSession(sess, self.graph, variables=self.variables) as sess:
            return super().save(filename, sess=sess, **kwargs)

