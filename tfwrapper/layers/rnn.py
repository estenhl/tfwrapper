import tensorflow as tf

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