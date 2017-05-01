import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.utils.exceptions import InvalidArgumentException

from .cnn import CNN

class SqueezeNet(CNN):
    def __init__(self, X_shape, classes, sess=None, name='SqueezeNet'):
        height, width, channels = X_shape

        if not (height == 224 and width == 224):
            raise InvalidArgumentException('Squeezenet currently only handles 224x224 images')

        layers = [
            self.conv2d(filter=[7, 7], depth=96, strides=2, name=name + '/conv1'),
            self.maxpool2d(k=3, strides=2, name=name + '/pool1'),
            self.fire(squeeze_depth=16, expand_depth=64, name=name + '/fire1'),
            self.fire(squeeze_depth=16, expand_depth=64, name=name + '/fire2'),
            self.fire(squeeze_depth=32, expand_depth=128, name=name + '/fire3'),
            self.maxpool2d(k=2, name=name + '/pool2'),
            self.fire(squeeze_depth=32, expand_depth=128, name=name + '/fire4'),
            self.fire(squeeze_depth=48, expand_depth=192, name=name + '/fire5'),
            self.fire(squeeze_depth=48, expand_depth=192, name=name + '/fire6'),
            self.fire(squeeze_depth=64, expand_depth=256, name=name + '/fire7'),
            self.maxpool2d(k=2, name=name + '/pool3'),
            self.fire(squeeze_depth=64, expand_depth=256, name=name + '/fire8'),
            self.avgpool2d(k=13, name=name + '/pool4'),
            self.out([7*7*512, classes], classes, name=name + '/pred')
        ]

        with TFSession(sess) as sess:
            super().__init__(X_shape, classes, layers, sess=sess, name=name)

    def fire(self, *, squeeze_depth, expand_depth, name='fire'):
        def create_layer(x):
            inputs = self.conv2d(filter=[1, 1], depth=squeeze_depth, name=name + '/squeeze')(x)
            expand_1x1 = self.conv2d(filter=[1, 1], depth=expand_depth, name=name + '/1x1')(inputs)
            expand_3x3 = self.conv2d(filter=[3, 3], depth=expand_depth, name=name + '/3x3')(inputs)

            return tf.concat([expand_1x1, expand_3x3], axis=3, name=name)

        return create_layer