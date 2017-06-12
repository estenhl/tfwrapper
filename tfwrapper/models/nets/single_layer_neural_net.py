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
        with TFSession(sess) as sess:
            X_size = np.prod(X_shape)

            layers = [
                fullyconnected(inputs=X_size, outputs=hidden, name=name + '/hidden'),
                out(inputs=hidden, outputs=y_size, name=name + '/pred')
            ]

            super().__init__(X_shape, y_size, layers, sess=sess, name=name)

            self.graph = sess.graph

    def save(self, filename, sess=None, **kwargs):
        with TFSession(sess, self.graph, variables=self.variables) as sess:
            kwargs['hidden'] = self.hidden
            super().save(filename, sess=sess, **kwargs)

    @classmethod
    def from_tw(cls, filename, sess=None, **kwargs):
        metadata_filename = '%s.%s' % (filename, 'tw')
        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)

        name = metadata['name']
        X_shape = metadata['X_shape']
        y_size = metadata['y_size']
        hidden = metadata['hidden']
        batch_size = metadata['batch_size']
        with TFSession(sess) as sess:

            net = SingleLayerNeuralNet(X_shape, y_size, hidden, name=name, sess=sess, **kwargs)
            net.load(filename)
            net.batch_size = batch_size

            return net
