import json
import numpy as np
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.layers import fullyconnected
from tfwrapper.layers import out

from .neural_net import NeuralNet

class SingleLayerNeuralNet(NeuralNet):
    init_args = {'hidden': 'hidden'}

    def __init__(self, X_shape, y_size, hidden, sess=None, name='SingleLayerNeuralNet'):
        self.hidden = hidden

        X_size = np.prod(X_shape)
        layers = [
            fullyconnected(inputs=X_size, outputs=hidden, name=name + '/hidden'),
            out(inputs=hidden, outputs=y_size, name=name + '/pred')
        ]
        with TFSession(sess) as sess:
            super().__init__(X_shape, y_size, layers, sess=sess, name=name)
            self.graph = sess.graph

    def save(self, filename, sess=None, **kwargs):
        with TFSession(sess, self.graph, variables=self.variables) as sess:
            kwargs['hidden'] = self.hidden
            return super().save(filename, sess=sess, **kwargs)
