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
from tfwrapper.layers import fullyconnected, out, recurring
from tfwrapper import Dataset
from tfwrapper.dataset import DatasetGenerator
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
                fullyconnected(inputs=X_size, outputs=hidden, name=name + '/hidden'),
                out(inputs=hidden, outputs=y_size, name=name + '/pred')
            ]

            return cls.from_shape(X_shape=X_shape, y_size=y_size, layers=layers, sess=sess, name=name)

    @classmethod
    def dual_layer(cls, X_shape, y_size, hidden1, hidden2, sess=None, name='DualLayerNeuralNet'):
        with TFSession(sess) as sess:
            X_size = np.prod(X_shape)

            layers = [
                fullyconnected(inputs=X_size, outputs=hidden1, name=name + '/hidden1'),
                fullyconnected(inputs=hidden1, outputs=hidden2, name=name + '/hidden2'),
                out(inputs=hidden2, outputs=y_size, name=name + '/pred')
            ]

            return cls.from_shape(X_shape=X_shape, y_size=y_size, layers=layers, sess=sess, name=name)

    @classmethod
    def recurrent(cls, seq_shape, seq_length, num_hidden, y_size, sess=None, name='RNN'):
        X_shape = seq_shape + [seq_length]
        layers = [recurring(seq_shape, seq_length, num_hidden, y_size, name=name)]

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

    def load(self, filename, sess=None):
        with TFSession(sess, self.graph, self.variables) as sess:
            super().load(filename, sess=sess)

            self.loss = sess.graph.get_tensor_by_name(self.name + '/loss:0')
            self.accuracy = sess.graph.get_tensor_by_name(self.name + '/accuracy:0')

    def train_epoch(self, generator, epoch_nr, feed_dict=None, val_generator=None, sess=None):
        if feed_dict is None:
            feed_dict = {}
        with TFSession(sess, self.graph) as sess:
            try:
                num_items = len(generator)
            except Exception as e:
                num_items = -1

            feed_dict[self.lr] = self.calculate_learning_rate(epoch=epoch_nr)

            epoch_loss_avg = 0
            epoch_acc_avg = 0
            epoch_time = 0

            item_count = 0
            batch_count = 0
            # Note: it's up to the Generator to shuffle on iter() start
            for batch_count, (batch_X, batch_y) in enumerate(generator):
                feed_dict[self.X] = batch_X
                feed_dict[self.y] = batch_y
                
                start_batch_time = process_time()
                _, loss_val, acc_val = sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=feed_dict)
                epoch_time += process_time() - start_batch_time

                epoch_loss_avg += loss_val
                epoch_acc_avg += acc_val

                item_count += len(batch_X)
                # TODO (11.05.17): Use logger
                if True:
                    # Display a summary of this batch reusing the same terminal line
                    sys.stdout.write('\033[K')  # erase previous line in case the new line is shorter
                    sys.stdout.write('\riter: {:0{}}/{} | batch loss: {:.5} - acc {:.5}' \
                                     ' | time: {:.3}s'.format(item_count, len(str(num_items)), num_items, 
                                                            loss_val, acc_val, epoch_time))
                    sys.stdout.flush()
            batch_count += 1
            epoch_loss_avg /= batch_count
            epoch_acc_avg /= batch_count

            # TODO (11.05.17): Use logger
            if True:
                epoch_summary = '\nEpoch: \033[1m\033[32m{}\033[0m\033[0m | avg batch loss: \033[1m\033[32m{:.5}\033[0m\033[0m' \
                                ' - avg acc: {:.5}'.format(epoch_nr + 1, epoch_loss_avg, epoch_acc_avg)
                
                if val_generator is not None and len(val_generator) > 0:
                    loss_val, acc_val = self.validate(generator=val_generator, sess=sess)
                    epoch_summary += ' | val_loss: \033[1m\033[32m{:.5}\033[0m\033[0m - val_acc: {:.5}'.format(loss_val, acc_val)
                print(epoch_summary, '\n')

    def validate(self, X=None, y=None, generator=None, feed_dict=None, sess=None, **kwargs):
        if X is not None and y is not None:
            generator = DatasetGenerator(Dataset(X=X, y=y), self.batch_size, shuffle=False)
        elif generator is None:
            errormsg = 'Either X and y or a generator must be supplied'
            logger.error(errormsg)
            raise InvalidArgumentException(errormsg)
        if generator.shuffle:
            logger.info('Disabling validation generator\'s shuffle')
            generator.shuffle = False

        feed_dict = self.parse_feed_dict(feed_dict, **kwargs)  # TODO (09.06.17) This isn't used. Why?
        with TFSession(sess, self.graph, variables=self.variables) as sess:
            preds = self.predict(generator=generator, sess=sess)
            y = y or np.concatenate([batch_y for _, batch_y in generator])
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

        net = NeuralNet.from_shape(X_shape, y_size, layers, name=name)

        for key in self.feed_dict:
            net.feed_dict[key] = self.feed_dict[key]

        for key in other.feed_dict:
            net.feed_dict[key] = other.feed_dict[key]

        return net
