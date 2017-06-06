import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.layers import conv2d, maxpool2d, fire, dropout, flatten, reshape
from tfwrapper.utils.exceptions import InvalidArgumentException

from .cnn import CNN

class SqueezeNet(CNN):
    def __init__(self, X_shape, classes, sess=None, name='SqueezeNet', version='1.1', **kwargs):
        keep_prob = tf.placeholder(tf.float32, name=name + '/dropout_placeholder')
        if version == '1.1':
            layers = [
                conv2d(filter=[3, 3], depth=64, strides=2, padding='VALID', init='xavier_normal', name=name + '/conv1'),
                maxpool2d(k=3, strides=2, name=name + '/pool1'),
                fire(squeeze_depth=16, expand_depth=64, name=name + '/fire2'),
                fire(squeeze_depth=16, expand_depth=64, name=name + '/fire3'),
                maxpool2d(k=3, strides=2, name=name + '/pool3'),
                fire(squeeze_depth=32, expand_depth=128, name=name + '/fire4'),
                fire(squeeze_depth=32, expand_depth=128, name=name + '/fire5'),
                maxpool2d(k=3, strides=2, name=name + '/pool5'),
                fire(squeeze_depth=48, expand_depth=192, name=name + '/fire6'),
                fire(squeeze_depth=48, expand_depth=192, name=name + '/fire7'),
                fire(squeeze_depth=64, expand_depth=256, name=name + '/fire8'),
                fire(squeeze_depth=64, expand_depth=256, name=name + '/fire9'),
                dropout(keep_prob, name=name + '/drop9'),
                conv2d(filter=[1, 1], depth=classes, strides=1, padding='VALID', init='xavier_normal', name=name + '/conv10'),
                flatten(name=name + 'avgpool10'),
                reshape([-1, classes], name=name + '/pred')
            ]
        elif version == '1.0':
            layers = [
                conv2d(filter=[7, 7], depth=96, strides=2, padding='VALID', init='xavier_normal', name=name + '/conv1'),
                maxpool2d(k=3, strides=2, name=name + '/pool1'),
                fire(squeeze_depth=16, expand_depth=64, name=name + '/fire2'),
                fire(squeeze_depth=16, expand_depth=64, name=name + '/fire3'),
                fire(squeeze_depth=32, expand_depth=128, name=name + '/fire4'),
                maxpool2d(k=3, strides=2, name=name + '/pool4'),
                fire(squeeze_depth=32, expand_depth=128, name=name + '/fire5'),
                fire(squeeze_depth=48, expand_depth=192, name=name + '/fire6'),
                fire(squeeze_depth=48, expand_depth=192, name=name + '/fire7'),
                fire(squeeze_depth=64, expand_depth=256, name=name + '/fire8'),
                maxpool2d(k=3, strides=2, name=name + '/pool8'),
                fire(squeeze_depth=64, expand_depth=256, name=name + '/fire9'),
                dropout(keep_prob, name=name + '/drop9'),
                conv2d(filter=[1, 1], depth=classes, strides=1, padding='VALID', init='xavier_normal', name=name + '/conv10'),
                flatten(name=name + 'avgpool10'),
                reshape([-1, classes], name=name + '/pred')
            ]
        else:
            raise NotImplementedError(
                '%s version not implemented (Valid: [\'1.0\', \'1.1\'])' % version)

        self.version = version
        with TFSession(sess) as sess:
            super().__init__(X_shape, classes, layers, sess=sess, name=name, **kwargs)

        self.feed_dict['keep_prob'] = {'placeholder': keep_prob, 'default': 1.}

    def save(self, filename, sess=None, **kwargs):
        return super().save(filename, sess=sess, version=self.version, **kwargs)
