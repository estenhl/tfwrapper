import tensorflow as tf

from tfwrapper import logger
from tfwrapper import TFSession
from tfwrapper.utils.exceptions import InvalidArgumentException

from .neural_net import NeuralNet

class CNN(NeuralNet):
    learning_rate = 0.001
    
    def __init__(self, X_shape, classes, layers, sess=None, name='NeuralNet'):
        with TFSession(sess) as sess:
            super().__init__(X_shape, classes, layers, sess=sess, name=name)