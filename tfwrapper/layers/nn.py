import tensorflow as tf

from .base import bias
from .base import weight

def fullyconnected(*, inputs, outputs, trainable=True, activation='relu', init='truncated', name='fullyconnected'):
    weight_shape = [1, 1, inputs, outputs]
    weight_name = name + '/weights'
    bias_name = name + '/biases'

    def create_layer(x):
        w = weight(weight_shape, name=weight_name, init=init, trainable=trainable)
        b = bias(outputs, name=bias_name, trainable=trainable)

        fc = tf.reshape(x, [-1, 1, 1, inputs], name=name + '/reshape')
        fc = tf.add(tf.matmul(fc, w), b, name=name + '/add')
        
        if activation == 'relu':
            fc = tf.nn.relu(fc, name=name)
        elif activation == 'softmax':
            fc = tf.nn.softmax(fc, name=name)
        else:
            raise NotImplementedError('%s activation is not implemented (Valid: [\'relu\', \'softmax\'])' % activation)

        return fc

    return create_layer


def dropout(dropout, name='dropout'):
    return lambda x: tf.nn.dropout(x, dropout, name=name)