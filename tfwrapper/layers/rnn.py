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