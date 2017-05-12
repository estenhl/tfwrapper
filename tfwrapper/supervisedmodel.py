import json
import math
import datetime
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from .logger import logger
from .tfsession import TFSession
from tfwrapper.dataset.dataset import batch_data
from tfwrapper.utils import get_variable_by_name
from tfwrapper.utils.exceptions import InvalidArgumentException

META_GRAPH_SUFFIX = 'meta'
METADATA_SUFFIX = 'tw'

class SupervisedModel(ABC):
    DEFAULT_BOTTLENECK_LAYER = -2

    graph = None
    variables = {}

    learning_rate = 0.1
    batch_size = 128

    # TODO: migrate constructor calls to .with_shape and remove method calls from __init__
    def __init__(self, X_shape=None, y_size=None, layers=None, preprocessing=[], sess=None, name='SupervisedModel'):
        self.name = name
        if X_shape is not None and y_size is not None and layers is not None:
            self.fill_from_shape(sess, X_shape, y_size, layers, preprocessing)
            self.post_init()

    def post_init(self):
        self.input_size = np.prod(self.X_shape)
        self.loss = self.loss_function()
        self.optimizer = self.optimizer_function()

    @classmethod
    def from_shape(cls, X_shape, y_size, layers, preprocessing=[], sess=None, name='SupervisedModel'):
        model = cls(X_shape=None, y_size=None, name=name)
        model.fill_from_shape(sess=sess, X_shape=X_shape, y_size=y_size, layers=layers)
        model.post_init()
        return model

    @classmethod
    def from_meta_graph(cls, filename, name):
        model = cls(X_shape=None, y_size=None, name=name)
        model.load_from_meta_graph(filename)
        model.post_init()
        return model

    def fill_from_shape(self, sess, X_shape, y_size, layers, preprocessing):
        with TFSession(sess) as sess:
            self.X_shape = X_shape
            self.input_size = np.prod(X_shape)
            self.y_size = y_size

            if type(y_size) is int:
                y_size = [y_size]

            self.X = tf.placeholder(tf.float32, [None] + X_shape, name=self.name + '/X_placeholder')
            self.y = tf.placeholder(tf.float32, [None] + y_size, name=self.name + '/y_placeholder')
            self.lr = tf.placeholder(tf.float32, [], name=self.name + '/learning_rate_placeholder')

            layers = preprocessing + layers
            
            self.layers = layers
            self.tensors = []
            prev = self.X
            for layer in layers:
                prev = layer(prev)
                self.tensors.append({'name': prev.name, 'tensor': prev})
            self.pred = prev

            self.graph = sess.graph
        self.init_vars_when_training = True
        self.feed_dict = {}

    def load_from_meta_graph(self, filename):
        with TFSession() as sess:
            metagraph_filename = '%s.%s' % (filename, META_GRAPH_SUFFIX)
            new_saver = tf.train.import_meta_graph(metagraph_filename, clear_devices=True)
            new_saver.restore(sess, metagraph_filename)

            graph = tf.get_default_graph()
            self.graph = graph
            self.X = graph.get_tensor_by_name(self.name + '/X_placeholder:0')
            self.y = graph.get_tensor_by_name(self.name + '/y_placeholder:0')
            self.lr = graph.get_tensor_by_name(self.name + '/learning_rate_placeholder:0')
            self.pred = graph.get_tensor_by_name(self.name + '/pred:0')

            self.X_shape = self.X.shape
            self.input_size = np.prod(self.X_shape)
            self.y_size = len(self.y)

            self.checkpoint_variables(sess)

    @abstractmethod
    def loss_function(self):
        raise NotImplementedError('SupervisedModel is a generic class')

    @abstractmethod
    def optimizer_function(self):
        raise NotImplementedError('SupervisedModel is a generic class')

    @staticmethod
    def bias(size, init='zeros', trainable=True, name='bias'):
        return SupervisedModel.weight([size], init=init, trainable=trainable, name=name)

    @staticmethod
    def weight(shape, init='truncated', stddev=0.02, trainable=True, name='weight'):
        if init == 'truncated':
            weight = tf.truncated_normal(shape, stddev=stddev)
        elif init == 'he_normal':
            # He et al., http://arxiv.org/abs/1502.01852
            fan_in, _ = SupervisedModel.compute_fan_in_out(shape)
            weight = tf.truncated_normal(shape, stddev=math.sqrt(2 / fan_in))
        elif init == 'xavier_normal':
            # Glorot & Bengio, AISTATS 2010 - http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
            fan_in, fan_out = SupervisedModel.compute_fan_in_out(shape)
            weight = tf.truncated_normal(shape, stddev=math.sqrt(2 / (fan_in + fan_out)))
        elif init == 'random':
            weight = tf.random_normal(shape)
        elif init == 'zeros':
            weight = tf.zeros(shape)
        else:
            raise NotImplementedError('Unknown initialization scheme %s' % str(init))

        return tf.Variable(weight, trainable=trainable, name=name)

    @staticmethod
    def compute_fan_in_out(weight_shape):
        if len(weight_shape) == 2:
            fan_in = weight_shape[0]
            fan_out = weight_shape[1]
        elif len(weight_shape) in {3, 4, 5}:
            # Assuming convolution kernels (1D, 2D or 3D).
            # TF kernel shape: (..., input_depth, depth)
            receptive_field_size = np.prod(weight_shape[:2])
            fan_in = weight_shape[-2] * receptive_field_size
            fan_out = weight_shape[-1] * receptive_field_size
        else:
            # No specific assumptions.
            fan_in = math.sqrt(np.prod(weight_shape))
            fan_out = math.sqrt(np.prod(weight_shape))
        return fan_in, fan_out

    @staticmethod
    def reshape(shape, name):
        return lambda x: tf.reshape(x, shape=shape, name=name)

    @staticmethod
    def out(*, inputs, outputs, init='truncated', trainable=True, name='pred'):
        weight_shape = [inputs, outputs]

        def create_layer(x):
            weight = SupervisedModel.weight(weight_shape, init=init, name=name + '/W', trainable=trainable)
            bias = SupervisedModel.bias(outputs, name=name + '/b')
            return tf.add(tf.matmul(x, weight), bias, name=name)

        return create_layer

    @staticmethod
    def relu(name):
        return lambda x: tf.nn.relu(x, name=name)

    @staticmethod
    def softmax(name):
        return lambda x: tf.nn.softmax(x, name=name)

    def run_op(self, target, *, source=None, data, feed_dict=None, sess=None):
        if type(target) in [str, int]:
            target = self.get_tensor(target)
        elif type(target) is not tf.Tensor:
            errormsg = 'Invalid type %s for target in run_op. (Valid is [\'str\', \'int\', \'tf.Tensor\'])' % repr(type(target))
            logger.error(errormsg)
            raise InvalidArgumentException(errormsg)

        if source is None:
            source = self.X

        if feed_dict is None:
            feed_dict = {}

        feed_dict[source] = data

        with TFSession(sess, self.graph) as sess:
            batches = batch_data(data, self.batch_size)
            result = None
            for batch in batches:
                feed_dict[source] = batch
                batch_result = sess.run(target, feed_dict=feed_dict)
                if result is None:
                    result = batch_result
                else:
                    result = np.concatenate([result, batch_result])

            return result

    def extract_features(self, data, layer, sess=None):
        if len(data.shape) == 3:
            data = np.reshape(data, (-1,) + data.shape)

        with TFSession(sess, self.graph) as sess:
            return self.run_op(layer, data=data, sess=sess)

    def extract_bottleneck_features(self, data, sess=None):
        with TFSession(sess, self.graph) as sess:
            features = self.extract_features(data, layer=self.DEFAULT_BOTTLENECK_LAYER, sess=sess)
            features = features.flatten()

            return features

    def checkpoint_variables(self, sess):
        for variable in tf.global_variables():
            self.variables[variable.name] = sess.run(variable)

    def create_batches(self, X, y, prefix=''):
        if not len(X) == len(y):
            errormsg = '%sX and %sy must be same length, not %d and %d' % (prefix, prefix, len(X), len(y))
            logger.error(errormsg)
            raise InvalidArgumentException(errormsg)

        if not (len(X.shape) >= 2 and list(X.shape[1:]) == self.X_shape):
            errormsg = '%sX with shape %s does not match given X_shape %s' % (prefix, str(X.shape), str(self.X_shape))
            logger.error(errormsg)
            raise InvalidArgumentException(errormsg)
        else:
            X = np.reshape(X, [-1] + self.X_shape)

        # TODO (11.05.17): This should not be in supervisedmodel (Makes regression on one variable impossible)
        if not len(y.shape) == 2:
            errormsg = '%sy must be a onehot array' % prefix
            logger.error(errormsg)
            raise InvalidArgumentException(errormsg)

        if not y.shape[1] == self.y_size:
            errormsg = '%sy with %d classes does not match given y_size %d' % (prefix, y.shape[1], self.y_size)
            logger.error(errormsg)
            raise InvalidArgumentException(errormsg)
        else:
            y = np.reshape(y, [-1, self.y_size])

        num_batches = int(len(X) / self.batch_size) + 1
        batches = []
        for i in range(num_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, len(X))
            batches.append((X[start:end], y[start:end]))

        return batches

    def parse_feed_dict(self, feed_dict, log=False, **kwargs):
        if feed_dict is None:
            feed_dict = {}

        for key in self.feed_dict:
            placeholder = self.feed_dict[key]['placeholder']
            value = self.feed_dict[key]['default']

            if key in kwargs:
                value = kwargs[key]
            elif log:
                logger.warning('Using default value %s for key %s. Set this value by passing a named parameter (e.g. net.train(..., %s=%s))' % (str(key), str(value), str(key), str(value)))

            feed_dict[placeholder] = value

        return feed_dict

    def calculate_learning_rate(self, **kwargs):
        if type(self.learning_rate) in [float, int]:
            return self.learning_rate
        elif callable(self.learning_rate):
            return self.learning_rate(**kwargs)
        else:
            errormsg = 'Invalid type %s for learning rate. (Valid is [\'float\', \'int\', \'func\'])' % type(self.learning_rate)
            logger.error(errormsg)
            raise InvalidArgumentException(errormsg)

    def assign_variable_value(self, name, value, sess=None):
        with TFSession(sess, self.graph) as sess:
            variable = get_variable_by_name(name)
            sess.run(variable.assign(value))

    def train(self, X=None, y=None, generator=None, feed_dict=None, epochs=None, val_X=None, val_y=None, val_generator=None, validate=True, shuffle=True, sess=None, **kwargs):
        feed_dict = self.parse_feed_dict(feed_dict, log=True, **kwargs)

        if X is not None and y is not None:
            logger.info('Training ' + self.name + ' with ' + str(len(X)) + ' cases')
            generator = self.create_batches(X, y, 'train.')
        elif generator is not None:
            logger.info('Training ' + self.name + ' with generator')
            shuffle = False
        else:
            errormsg = 'Either X and y or a generator must be supplied'
            logger.error(errormsg)
            raise InvalidArgumentException(errormsg)

        if epochs is None:
            logger.warning('Training without specifying epochs. Defaulting to 1')
            epochs = 1

        if val_X is not None and val_y is not None:
            val_generator = self.create_batches(val_X, val_y, 'val.')
        elif validate is not False and val_generator is None:
            if type(validate) is float:
                test_split = validate
            else:
                test_split = 0.2
            train_split = 1. - test_split
            try:
                train_len = max(int(len(generator) * train_split), 1)
                val_generator = generator[train_len:]
                generator = generator[:train_len]
            except Exception:
                val_generator = None
                logger.warning('Unable to split dataset into train and val when generator has no len')

        with TFSession(sess, self.graph, init=self.init_vars_when_training) as sess:
            for epoch in range(epochs):
                self.train_epoch(generator, epoch, feed_dict=feed_dict, val_batches=val_generator, shuffle=shuffle, sess=sess)

            self.checkpoint_variables(sess)

    @abstractmethod
    def train_epoch(self, generator, epoch_nr, feed_dict={}, val_batches=None, sess=None):
        raise NotImplementedError('SupervisedModel is a generic class')

    def predict(self, X, feed_dict=None, sess=None, **kwargs):
        feed_dict = self.parse_feed_dict(feed_dict, **kwargs)

        with TFSession(sess, self.graph, variables=self.variables) as sess:
            batches = batch_data(X, self.batch_size)
            preds = None

            for batch in batches:
                feed_dict[self.X] = batch
                batch_preds = sess.run(self.pred, feed_dict=feed_dict)
                if preds is not None:
                    preds = np.concatenate([preds, batch_preds])
                else:
                    preds = batch_preds

        return preds

    @abstractmethod
    def validate(self, X, y, feed_dict={}, sess=None, **kwargs):
        raise NotImplementedError('SupervisedModel is a generic class')

    def save(self, filename, sess=None, **kwargs):
        with TFSession(sess, self.graph, variables=self.variables) as sess:
            saver = tf.train.Saver()
            saver.save(sess, filename, meta_graph_suffix=META_GRAPH_SUFFIX)

            metadata = kwargs
            metadata['name'] = self.name
            metadata['X_shape'] = self.X_shape
            metadata['y_size'] = self.y_size
            metadata['batch_size'] = self.batch_size
            metadata['time'] = str(datetime.datetime.now())

            metadata_filename = '%s.%s' % (filename, METADATA_SUFFIX)
            with open(metadata_filename, 'w') as f:
                f.write(json.dumps(metadata, indent=2))

    def load(self, filename, sess=None):
        with TFSession(sess, sess.graph) as sess:
            graph_path = filename + '.meta'
            saver = tf.train.Saver()
            saver.restore(sess, filename)

            self.graph = sess.graph
            self.X = sess.graph.get_tensor_by_name(self.name + '/X_placeholder:0')
            self.y = sess.graph.get_tensor_by_name(self.name + '/y_placeholder:0')
            self.lr = sess.graph.get_tensor_by_name(self.name + '/learning_rate_placeholder:0')
            self.pred = sess.graph.get_tensor_by_name(self.name + '/pred:0')

            self.checkpoint_variables(sess)

            # TODO (11.05.17): SHOULD USE METADATA, NOT SURE HOW THOUGH

    def get_tensor(self, val):
        if type(val) is int:
            return self.tensors[val]['tensor']
        elif type(val) is str:
            for i in range(len(self.tensors)):
                if self.tensors[i]['name'] == val:
                    return self.tensors[i]['tensor']

            errormsg = 'Invalid tensor name %s' % val
        else:
            errormsg = 'Invalid type %s for get_tensor. (Valid is [\'int\', \'str\'])' % repr(type(val))

        logger.error(errormsg)
        raise InvalidArgumentException(errormsg)

    def __len__(self):
        return len(self.tensors)
