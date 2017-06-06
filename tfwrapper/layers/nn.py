import tensorflow as tf

from .base import bias
from .base import weight

def fullyconnected(X=None, *, inputs, outputs, trainable=True, activation='relu', init='truncated', name='fullyconnected'):
    if X is None:
        return lambda x: fullyconnected(X=x, inputs=inputs, outputs=outputs, trainable=trainable, activation=activation, init=init, name=name)

    weight_shape = [inputs, outputs]
    weight_name = name + '/W'
    bias_name = name + '/b'


    W = weight(weight_shape, name=weight_name, init=init, trainable=trainable)
    b = bias(outputs, name=bias_name, trainable=trainable)

    fc = tf.reshape(X, [-1, inputs], name=name + '/reshape')
    fc = tf.add(tf.matmul(fc, W), b, name=name + '/add')
    
    if activation == 'relu':
        return tf.nn.relu(fc, name=name)
    elif activation == 'softmax':
        return tf.nn.softmax(fc, name=name)

    raise NotImplementedError('%s activation is not implemented (Valid: [\'relu\', \'softmax\'])' % activation)

def dropout(X=None, *, keep_prob, name='dropout'):
    if X is None:
        return lambda x: dropout(X=x, keep_prob=keep_prob, name=name)

    return tf.nn.dropout(X, keep_prob, name=name)