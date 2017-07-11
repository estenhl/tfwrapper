import json
import math
import datetime
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from tfwrapper import logger
from tfwrapper import TFSession
from tfwrapper.dataset import Dataset
from tfwrapper.dataset.dataset import batch_data
from tfwrapper.dataset.dataset_generator import DatasetGenerator
from tfwrapper.dataset.dataset_generator import DatasetGeneratorBase
from tfwrapper.dataset.dataset_generator import GeneratorWrapper
from tfwrapper.utils import get_variable_by_name
from tfwrapper.utils.exceptions import raise_exception
from tfwrapper.utils.exceptions import InvalidArgumentException
from tfwrapper.utils.data import get_subclass_by_name

from tfwrapper import logger
from tfwrapper import TFSession
from tfwrapper import METADATA_SUFFIX
from tfwrapper.dataset.dataset import batch_data
from tfwrapper.utils import get_variable_by_name
from tfwrapper.layers import Layer


META_GRAPH_SUFFIX = 'meta'


class SupervisedModel(ABC):
    DEFAULT_BOTTLENECK_LAYER = -2

    learning_rate = 0.1
    batch_size = 128

    # TODO: migrate constructor calls to .with_shape and remove method calls from __init__
    def __init__(self, X_shape=None, y_size=None, layers=None, preprocessing=None, sess=None, name='SupervisedModel'):
        if preprocessing is None:
            preprocessing = []
            
        self.name = name

        if X_shape is not None and y_size is not None and layers is not None:
            with TFSession(sess) as sess:
                self.fill_from_shape(X_shape, y_size, layers, preprocessing, sess=sess)
                self.post_init()

    def post_init(self):
        self.input_size = np.prod(self.X_shape)
        self.loss = self.loss_function()
        self.optimizer = self.optimizer_function()

    @classmethod
    def from_shape(cls, X_shape, y_size, layers, preprocessing=None, sess=None, name='SupervisedModel'):
        if preprocessing is None:
            preprocessing = []

        with TFSession(sess) as sess:
            model = cls(X_shape=None, y_size=None, name=name, sess=sess)
            model.fill_from_shape(sess=sess, X_shape=X_shape, y_size=y_size, layers=layers, preprocessing=preprocessing)
            model.post_init()

        return model

    @classmethod
    def from_meta_graph(cls, filename, name):
        with TFSession() as sess:
            model = cls(X_shape=None, y_size=None, name=name, sess=sess)
            model.load_from_meta_graph(filename)
            model.post_init()

        return model

    @classmethod
    def from_tw(cls, filename, sess=None, **kwargs):
        if filename.endswith('.tw'):
            metadata_filename = filename
            weights_filename = filename[:-3]
        else:
            metadata_filename = '%s.%s' % (filename, 'tw')
            weights_filename = filename

        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)

        name = metadata['name']
        X_shape = metadata['X_shape']
        y_size = metadata['y_size']
        batch_size = metadata['batch_size']
        classname = metadata['type']

        from .nets import SingleLayerNeuralNet
        subclass = get_subclass_by_name(cls, classname)

        for key in subclass.init_args:
            value = subclass.init_args[key]

            if value in metadata:
                kwargs[key] = metadata[value]
            else:
                logger.warning('Trying to fetch non-existing pair (%s, %s) from tw file %s' % (key, value, filename))

        with TFSession(sess) as sess:
            net = subclass(X_shape, y_size, name=name, sess=sess, **kwargs)
            net.load(weights_filename, sess=sess)
            net.batch_size = batch_size

            return net

    def fill_from_shape(self, X_shape, y_size, layers, preprocessing=None, sess=None):
        if preprocessing is None:
            preprocessing = []

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
                if type(layer) is Layer:
                    if layer.dependencies is not None:
                        dependencies = layer.dependencies

                        if type(dependencies) is str:
                            prev = sess.graph.get_tensor_by_name('/'.join([self.name, dependencies]) + ':0')
                        elif type(dependencies) is list:
                            prev = [sess.graph.get_tensor_by_name('/'.join([self.name, dependency]) + ':0') for dependency in dependencies]
                        else:
                            raise_exception('Invalid layer dependency type %s. (Valid is [str, list])' % type(dependencies), InvalidArgumentException)
                prev = layer(prev)
                self.tensors.append({'name': prev.name, 'tensor': prev})
            self.pred = prev

            self.graph = sess.graph

        self.variables = {}
        self.init_vars_when_training = True
        self.feed_dict = {}

    def load_from_meta_graph(self, filename, sess=None):
        with TFSession(sess) as sess:
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

    def reset(self):
        with TFSession(None, self.graph) as sess:
            sess.run(tf.global_variables_initializer())

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
        for variable in tf.trainable_variables():
            self.variables[variable.name] = {'tensor': variable, 'value': sess.run(variable)}

    def validate_batches(self, X, y, prefix=''):
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

    def parse_feed_dict(self, feed_dict, log=False, **kwargs):
        if feed_dict is None:
            feed_dict = {}

        for key in self.feed_dict:
            placeholder = self.feed_dict[key]['placeholder']
            value = self.feed_dict[key]['default']

            if key in kwargs:
                value = kwargs[key]
            elif log:
                logger.warning('Using default value %s for key %s. Set this value by passing a keyword parameter (e.g. net.train(..., %s=%s))' % (repr(value), repr(key), str(key), repr(value)))

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
            logger.info('Training %s with %d cases' % (self.name, len(X)))
            # Create a basic sample generator for the provided data
            self.validate_batches(X, y)
            generator = DatasetGenerator(Dataset(X=X, y=y), self.batch_size, shuffle=shuffle)
        elif generator is not None:
            if not issubclass(generator, DatasetGeneratorBase):
                generator = GeneratorWrapper(generator)
            logger.info('Training %s with generator' % self.name)
            if not generator.shuffle and shuffle:
                logger.warning('Your specified generator is not set to shuffle, yet shuffle was specified for training')
        else:
            errormsg = 'Either X and y or a generator must be supplied'
            logger.error(errormsg)
            raise InvalidArgumentException(errormsg)

        if epochs is None:
            logger.warning('Training without specifying epochs. Defaulting to 1')
            epochs = 1

        if val_X is not None and val_y is not None:
            if generator.normalize:
                # Generator is doing batch normalization. Val expects a normalized value, but we need to avoid batching
                val_generator = DatasetGenerator(Dataset(X=val_X, y=val_y), 1, shuffle=False, normalize=True)
            else:
                # Don't know whether the provided validation data is pre-normalized.
                # Normalize==False won't change its state, hence it's safe to do validation in batches either way
                val_generator = DatasetGenerator(Dataset(X=val_X, y=val_y), self.batch_size, shuffle=False, normalize=False)

        elif validate is not False and val_generator is None:
            if type(validate) is float:
                test_split = validate
            else:
                test_split = 0.2
            train_split = 1. - test_split
            try:
                train_len = max(int(generator.get_base_length() * train_split), 1)
                val_generator = generator[train_len:]
                generator = generator[:train_len]
            except Exception:
                val_generator = None
                logger.warning('Unable to split dataset into train and val when generator has no len')

        with TFSession(sess, self.graph, init=self.init_vars_when_training) as sess:
            for epoch in range(epochs):
                self.train_epoch(generator, epoch, feed_dict=feed_dict, val_generator=val_generator, sess=sess)

            self.checkpoint_variables(sess)

    @abstractmethod
    def train_epoch(self, generator, epoch_nr, feed_dict=None, val_generator=None, sess=None):
        raise NotImplementedError('SupervisedModel is a generic class')

    def predict(self, X=None, generator=None, feed_dict=None, sess=None, **kwargs):
        if X is not None:
            generator = DatasetGenerator(Dataset(X=X, y=np.full(len(X), -1)), self.batch_size, shuffle=False, normalize=False)
        elif generator is None:
            errormsg = 'Either X and y or a generator must be supplied'
            logger.error(errormsg)
            raise InvalidArgumentException(errormsg)
        if generator.shuffle:
            logger.warning('Disabling validation generator\'s shuffle')
            generator.shuffle = False
        if generator.normalize and generator.batch_size > 1:
            logger.warning('Reducing validation generator batch size due to its batch normalization.')
            generator.batch_size = 1

        feed_dict = self.parse_feed_dict(feed_dict, **kwargs)
        with TFSession(sess, self.graph, variables=self.variables) as sess:
            preds = None

            for batch_X, _ in generator:
                feed_dict[self.X] = batch_X
                batch_preds = sess.run(self.pred, feed_dict=feed_dict)
                if preds is not None:
                    preds = np.concatenate([preds, batch_preds])
                else:
                    preds = batch_preds

        return preds

    @abstractmethod
    def validate(self, X=None, y=None, generator=None, feed_dict=None, sess=None, **kwargs):
        raise NotImplementedError('SupervisedModel is a generic class')

    def save(self, filename, sess=None, **kwargs):
        with TFSession(sess, self.graph, variables=self.variables) as sess:
            saver = tf.train.Saver(tf.trainable_variables())
            saver.save(sess, filename, meta_graph_suffix=META_GRAPH_SUFFIX)

            metadata = kwargs
            metadata['name'] = self.name
            metadata['X_shape'] = self.X_shape
            metadata['y_size'] = self.y_size
            metadata['batch_size'] = self.batch_size
            metadata['time'] = str(datetime.datetime.now())
            metadata['type'] = self.__class__.__name__

            metadata_filename = '%s.%s' % (filename, METADATA_SUFFIX)
            with open(metadata_filename, 'w') as f:
                f.write(json.dumps(metadata, indent=2))

    def load(self, filename, sess=None):
        with TFSession(sess, self.graph) as sess:
            graph_path = filename + '.meta'
            saver = tf.train.Saver(tf.trainable_variables())
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
