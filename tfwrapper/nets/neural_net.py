import sys
import copy
import random
import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from time import process_time

from tfwrapper import logger
from tfwrapper import TFSession
from tfwrapper import SupervisedModel
from tfwrapper.utils.exceptions import InvalidArgumentException

class NeuralNet(SupervisedModel):
    def __init__(self, X_shape=None, y_size=None, layers=None, sess=None, name='NeuralNet', **kwargs):
        super().__init__(name=name)
        if X_shape is not None and y_size is not None and layers is not None:
            with TFSession(sess) as sess:
                self.fill_from_shape(sess, X_shape, y_size, layers, **kwargs)
                self.post_init()

    def post_init(self):
        super().post_init()
        self.accuracy = self.accuracy_function()

    @classmethod
    def single_layer(cls, X_shape, y_size, hidden, sess=None, name='SingleLayerNeuralNet'):
        with TFSession(sess) as sess:
            X_size = np.prod(X_shape)

            layers = [
                cls.fullyconnected(inputs=X_size, outputs=hidden, name=name + '/hidden'),
                cls.out(inputs=hidden, outputs=y_size, name=name + '/pred')
                ]

            return cls.from_shape(X_shape=X_shape, y_size=y_size, layers=layers, sess=sess, name=name)

    @classmethod
    def dual_layer(cls, X_shape, y_size, hidden1, hidden2, sess=None, name='DualLayerNeuralNet'):
        with TFSession(sess) as sess:
            X_size = np.prod(X_shape)

            layers = [
                cls.fullyconnected(inputs=X_size, outputs=hidden1, name=name + '/hidden1'),
                cls.fullyconnected(inputs=hidden1, outputs=hidden2, name=name + '/hidden2'),
                cls.out(inputs=hidden2, outputs=y_size, name=name + '/pred')
            ]

            return cls.from_shape(X_shape=X_shape, y_size=y_size, layers=layers, sess=sess, name=name)

    @classmethod
    def rnn(cls, seq_shape, seq_length, num_hidden, y_size, sess=None, name='RNN'):
        X_shape = seq_shape + [seq_length]
        layers = [cls.rnn_layer(seq_shape, seq_length, num_hidden, y_size, name=name)]

        return cls.from_shape(X_shape=X_shape, y_size=y_size, layers=layers, sess=sess, name=name)

    def load_from_meta_graph(self, filename):
        super().load_from_meta_graph(filename=filename)
        self.loss = self.graph.get_tensor_by_name(self.name + '/loss:0')
        self.accuracy = self.graph.get_tensor_by_name(self.name + '/accuracy:0')

    def loss_function(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y, name=self.name + '/softmax'), name=self.name + '/loss')

    def optimizer_function(self):
        return tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-5, name=self.name + '/adam').minimize(self.loss, name=self.name + '/optimizer')

    def accuracy_function(self):
        correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32), name=self.name + '/accuracy')

    @staticmethod
    def fullyconnected(*, inputs, outputs, trainable=True, activation='relu', init='truncated', name='fullyconnected'):
        weight_shape = [inputs, outputs]
        weight_name = name + '/W'
        bias_name = name + '/b'

        def create_layer(x):
            weight = NeuralNet.weight(weight_shape, name=weight_name, init=init, trainable=trainable)
            bias = NeuralNet.bias(outputs, name=bias_name, trainable=trainable)

            fc = tf.reshape(x, [-1, inputs], name=name + '/reshape')
            fc = tf.add(tf.matmul(fc, weight), bias, name=name + '/add')

            if activation == 'relu':
                fc = tf.nn.relu(fc, name=name)
            elif activation == 'softmax':
                fc = tf.nn.softmax(fc, name=name)
            else:
                raise NotImplementedError('%s activation is not implemented (Valid: [\'relu\', \'softmax\'])' % activation)

            return fc

        return create_layer

    @staticmethod
    def dropout(dropout, name='dropout'):
        return lambda x: tf.nn.dropout(x, dropout, name=name)

    @staticmethod
    def rnn_layer(seq_shape, seq_length, num_hidden, classes, name):
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

    def load(self, filename, sess=None):
        with TFSession(sess, self.graph, self.variables) as sess:
            super().load(filename, sess=sess)

            self.loss = sess.graph.get_tensor_by_name(self.name + '/loss:0')
            self.accuracy = sess.graph.get_tensor_by_name(self.name + '/accuracy:0')

    def train_epoch(self, batches, epoch_nr, feed_dict={}, val_batches=None, shuffle=False, sess=None):
        with TFSession(sess, self.graph) as sess:
            # TODO (11.05.17): Generators has noe __len__
            try:
                num_batches = len(batches)
                num_items = num_batches * self.batch_size - (self.batch_size - len(batches[-1][0]))
            except Exception as e:
                num_items = -1

            if shuffle:
                batches = batches.copy()
                random.shuffle(batches)
            
            feed_dict[self.lr] = self.calculate_learning_rate(epoch=epoch_nr)

            epoch_loss_avg = 0
            epoch_acc_avg = 0
            epoch_time = 0

            batch_count = 0
            item_count = 0
            for X, y in batches:
                feed_dict[self.X] = X
                feed_dict[self.y] = y
                
                start_batch_time = process_time()
                _, loss_val, acc_val = sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=feed_dict)
                epoch_time += process_time() - start_batch_time

                epoch_loss_avg += loss_val
                epoch_acc_avg += acc_val

                batch_count += 1
                item_count += len(X)
                # TODO (11.05.17): Use logger
                if True:
                    # Display a summary of this batch reusing the same terminal line
                    sys.stdout.write('\033[K')  # erase previous line in case the new line is shorter
                    sys.stdout.write('\riter: {:0{}}/{} | batch loss: {:.5} - acc {:.5}' \
                                     ' | time: {:.3}s'.format(item_count, len(str(num_items)), num_items, 
                                                            loss_val, acc_val, epoch_time))
                    sys.stdout.flush()

            epoch_loss_avg /= batch_count
            epoch_acc_avg /= batch_count

            # TODO (11.05.17): Use logger
            if True:
                epoch_summary = '\nEpoch: \033[1m\033[32m{}\033[0m\033[0m | avg batch loss: \033[1m\033[32m{:.5}\033[0m\033[0m' \
                                ' - avg acc: {:.5}'.format(epoch_nr + 1, epoch_loss_avg, epoch_acc_avg)
                
                # TODO (11.05.17): Handle batches as actual batches
                if val_batches is not None and len(val_batches) > 0:
                    X = None
                    y = None

                    for batch_X, batch_y in val_batches:
                        if X is None:
                            X = batch_X
                        else:
                            X = np.concatenate([X, batch_X])

                        if y is None:
                            y = batch_y
                        else:
                            y = np.concatenate([y, batch_y])

                    X = np.asarray(X)
                    y = np.asarray(y)
                    loss_val, acc_val = self.validate(X, y, sess=sess)
                    epoch_summary += ' | val_loss: \033[1m\033[32m{:.5}\033[0m\033[0m - val_acc: {:.5}'.format(loss_val, acc_val)
                print(epoch_summary, '\n')

    def validate(self, X, y, feed_dict=None, sess=None, **kwargs):
        feed_dict = self.parse_feed_dict(feed_dict, **kwargs)
        with TFSession(sess, self.graph, variables=self.variables) as sess:
            preds = self.predict(X, sess=sess)
            loss, acc = sess.run([self.loss, self.accuracy], feed_dict={self.pred: preds, self.y: y})
        
        return loss, acc

    def __add__(self, other):
        layers = self.layers + other.layers
        name = '%s_%s' % (self.name, other.name)
        X_shape = self.X_shape
        y_size = other.y_size

        if not self.y_size == other.X_shape:
            errormsg = 'Unable to join neural nets with last layer shape %s and first layer shape %s' % (str(X_shape), str(y_size))
            logger.error(errormsg)
            raise InvalidArgumentException(errormsg)

        net = NeuralNet(X_shape, y_size, layers, name=name)

        for key in self.feed_dict:
            net.feed_dict[key] = self.feed_dict[key]

        for key in other.feed_dict:
            net.feed_dict[key] = other.feed_dict[key]

        return net
