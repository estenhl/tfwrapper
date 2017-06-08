import tensorflow as tf

from tfwrapper import logger
from tfwrapper import TFSession
from tfwrapper.layers import conv2d, dropout, fullyconnected, maxpool2d, out, relu, reshape, softmax
from tfwrapper.utils.exceptions import InvalidArgumentException

from .neural_net import NeuralNet

class CNN(NeuralNet):
    learning_rate = 0.001

    def __init__(self, X_shape=None, y_size=None, layers=None, sess=None, name='NeuralNet', **kwargs):
        super().__init__(name=name)
        if X_shape is not None and y_size is not None and layers is not None:
            with TFSession(sess) as sess:
                self.fill_from_shape(sess, X_shape, y_size, layers, **kwargs)
                self.post_init()

    @classmethod
    def shallow(cls, X_shape, y_size, sess=None, name='ShallowCNN'):
        height, width, channels = X_shape
        twice_reduce = lambda x: -(-x // 4)
        fc_input_size = twice_reduce(height) * twice_reduce(width) * 64

        layers = [
            reshape([-1, height, width, channels], name=name + '/reshape'),
            conv2d(filter=[5, 5], depth=32, name=name + '/conv1'),
            maxpool2d(k=2, name=name + '/pool1'),
            conv2d(filter=[5, 5], depth=64, name=name + '/conv2'),
            conv2d(filter=[5, 5], depth=64, name=name + '/conv3'),
            maxpool2d(k=2, name=name + '/pool2'),
            fullyconnected(inputs=fc_input_size, outputs=512, name=name + '/fc'),
            dropout(0.8, name=name + '/dropout'),
            out(inputs=512, outputs=y_size, name=name + '/pred')
        ]

        with TFSession(sess) as sess:
            return cls.from_shape(X_shape=X_shape, y_size=y_size, layers=layers, sess=sess, name=name)

    @classmethod
    def vgg16(cls, X_shape, y_size=1000, sess=None, name='VGG16'):
        height, width, channels = X_shape

        if not ((height % 2 ** 5) == 0 and (width % 2 ** 5) == 0):
            raise ValueError('Height and width must be divisible by %d' % 2 ** 5)

        fc_input_size = int(height / (2**5)) * int(width / (2**5)) * 512

        layers = [
            conv2d(filter=[3, 3], depth=64, name=name + '/conv1/conv1_1'),
            conv2d(filter=[3, 3], depth=64, name=name + '/conv1/conv1_2'),
            maxpool2d(k=2),
            conv2d(filter=[3, 3], depth=128, name=name + '/conv2/conv2_1'),
            conv2d(filter=[3, 3], depth=128, name=name + '/conv2/conv2_2'),
            maxpool2d(k=2),
            conv2d(filter=[3, 3], depth=256, name=name + '/conv3/conv3_1'),
            conv2d(filter=[3, 3], depth=256, name=name + '/conv3/conv3_2'),
            conv2d(filter=[3, 3], depth=256, name=name + '/conv3/conv3_3'),
            maxpool2d(k=2),
            conv2d(filter=[3, 3], depth=512, name=name + '/conv4/conv4_1'),
            conv2d(filter=[3, 3], depth=512, name=name + '/conv4/conv4_2'),
            conv2d(filter=[3, 3], depth=512, name=name + '/conv4/conv4_3'),
            maxpool2d(k=2),
            conv2d(filter=[3, 3], depth=512, name=name + '/conv5/conv5_1'),
            conv2d(filter=[3, 3], depth=512, name=name + '/conv5/conv5_2'),
            conv2d(filter=[3, 3], depth=512, name=name + '/conv5/conv5_3'),
            maxpool2d(k=2),
            fullyconnected(inputs=fc_input_size, outputs=4096, name=name + '/fc6'),
            relu(name=name + '/relu1'),
            dropout(0.5),
            fullyconnected(inputs=4096, outputs=4096, name=name + '/fc7'),
            relu(name=name + '/relu2'),
            dropout(0.5),
            fullyconnected(inputs=4096, outputs=y_size, name=name + '/fc8'),
            softmax(name=name + '/pred')
        ]

        return cls.from_shape(X_shape=X_shape, y_size=y_size, layers=layers, sess=sess, name=name)
