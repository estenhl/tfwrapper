from .neural_net import NeuralNet

class RNN(NeuralNet):
    def __init__(self, seq_shape, seq_length, num_hidden, classes, sess=None, name='RNN'):
        raise NotImplementedError('RNN is not implemented!')
