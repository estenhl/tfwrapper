import sys
import copy
import random
import numpy as np
import tensorflow as tf

from time import process_time
from tfwrapper import TFSession
from tfwrapper import SupervisedModel

class NeuralNet(SupervisedModel):
    def __init__(self, X_shape, classes, layers, sess=None, name='NeuralNet'):

        with TFSession(sess) as sess:
            super().__init__(X_shape, classes, layers, sess=sess, name=name)

            self.accuracy = self.accuracy_function()

            self.graph = sess.graph

    def loss_function(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y, name=self.name + '/softmax'), name=self.name + '/loss')

    def optimizer_function(self):
        return tf.train.AdamOptimizer(learning_rate=self.lr, name=self.name + '/adam').minimize(self.loss, name=self.name + '/optimizer')

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
            
            feed_dict[self.lr] = self.learning_rate

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

                    for batch_X, batch_y in val_batches[1:]:
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

    def validate(self, X, y, sess=None):
        with TFSession(sess, self.graph, variables=self.variables) as sess:
            preds = self.predict(X, sess=sess)
            loss, acc = sess.run([self.loss, self.accuracy], feed_dict={self.pred: preds, self.y: y})
        
        return loss, acc
