import tensorflow as tf

from .cnn import conv2d

def fire(*, squeeze_depth, expand_depth, name='fire'):
    def create_layer(x):
        inputs = conv2d(filter=[1, 1], depth=squeeze_depth, padding='VALID', init='xavier_normal', name=name + '/squeeze')(x)
        expand_1x1 = conv2d(filter=[1, 1], depth=expand_depth, padding='VALID', init='xavier_normal', name=name + '/1x1')(inputs)
        expand_3x3 = conv2d(filter=[3, 3], depth=expand_depth, padding='SAME', init='xavier_normal', name=name + '/3x3')(inputs)

        return tf.concat([expand_1x1, expand_3x3], axis=3, name=name)

    return create_layer