import tensorflow as tf
from tensorflow.contrib import rnn

def recurring(seq_shape, seq_length, num_hidden, classes, name):
    def create_layer(x):
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1] + seq_shape)
        x = tf.split(x, seq_length, 0)

        lstm_cell = rnn.BasicLSTMCell(128, forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        weight = tf.Variable(tf.random_normal([num_hidden, classes]), name=name + '_W')
        bias = tf.Variable(tf.random_normal([classes]), name=name + '_b')
        return tf.add(tf.matmul(outputs[-1], weight), bias, name=name + '/pred')

    return create_layer

def lstm_layer(hidden, y_size):
    def create_layer(X):
        cell = tf.nn.rnn_cell.LSTMCell(hidden, state_is_tuple=True)

        val, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)

        weight = tf.Variable(tf.truncated_normal([hidden, y_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[y_size]))

        return tf.nn.softmax(tf.matmul(last, weight) + bias)
    return create_layer