import json
import datetime
import numpy as np
import tensorflow as tf
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from typing import Callable, Dict, List, Union

from tfwrapper import TFSession, METADATA_SUFFIX
from tfwrapper.layers import Layer
from tfwrapper.layers.accuracy import Accuracy, CorrectPred
from tfwrapper.layers.loss import Loss, MeanSoftmaxCrossEntropy
from tfwrapper.layers.optimizers import Optimizer, Adam
from tfwrapper.utils.data import get_subclass_by_name
from tfwrapper.utils.exceptions import log_and_raise, InvalidArgumentException


META_GRAPH_SUFFIX = 'meta'


_tensor_wrapper = Callable[[Union[tf.Tensor, List[tf.Tensor]]], tf.Tensor]


def _parse_layer_list(start: tf.Tensor, remaining: List[Union[tf.Tensor, _tensor_wrapper]], sess: tf.Session = None):
    tensors = [{'name': start.name, 'tensor': start}]

    prev = start
    for layer in remaining:
        if type(layer) is Layer:
            if layer.dependencies is not None:
                dependencies = layer.dependencies

                if type(dependencies) is str:
                    prev = sess.graph.get_tensor_by_name('/'.join([self.name, dependencies]) + ':0')
                elif type(dependencies) is list:
                    prev = [sess.graph.get_tensor_by_name('/'.join([self.name, dependency]) + ':0') for dependency in dependencies]
                else:
                    log_and_raise(InvalidArgumentException, 'Invalid layer dependency type %s. (Valid is [str, list])' % type(dependencies))
        elif not callable(layer):
            log_and_raise(InvalidArgumentException, 'Invalid layer typ %s. (Valid is [Layer, callable]' % str(type(layer)))

        prev = layer(prev)
        tensors.append({'name': prev.name, 'tensor': prev})
        
    return tensors, prev


class BaseModel(ABC):
    """ The base class for singular machine learning models. Mainly contains infrastructure stuff """

    def graph(self) -> tf.Graph:
        """ The tf.Graph containing all tensors and variables necessary for the model """
        return self._graph

    def variables(self) -> Dict[str, tf.Tensor]:
        """ A map of all the variable values stored in the tensors of the graph. Typically trained weight and bias values.
        Used for training the model in one session, then restoring the model on a separate session """
        return self._variables
    
    def __init__(self, X_shape: List[int], y_shape: Union[int, List[int]], layers: List[Union[tf.Tensor, _tensor_wrapper]], *, preprocessing: List[Union[tf.Tensor, _tensor_wrapper]] = None, sess: tf.Session = None, name: str = 'BaseModel'):
        if preprocessing is None:
            preprocessing = []

        self.name = name
        self.X_shape = X_shape
        if type(y_shape) is int:
            self.y_shape = [y_shape]
        elif type(y_shape) is list:
            self.y_shape = y_shape
        else:
            log_and_raise(InvalidArgumentException, 'Invalid type %s for y_size. (Valid is [int, list])' % str(type(y_size)))

        with TFSession(sess) as sess:
            self.X = tf.placeholder(tf.float32, [None] + self.X_shape, name=self.name + '/X_placeholder')
            self.y = tf.placeholder(tf.float32, [None] + self.y_shape, name=self.name + '/y_placeholder')
            self.lr = tf.placeholder(tf.float32, [], name=self.name + '/learning_rate_placeholder')

            self.layers = preprocessing + layers
            self.tensors, self.preds = _parse_layer_list(self.X, self.layers)
            self._graph = sess.graph

        self._variables = {}
        self.init_vars_when_training = True
        self.feed_dict = {}

    @classmethod
    def from_tw(cls, filename: str, sess: tf.Session = None, **kwargs):
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
        y_shape = metadata['y_shape']
        batch_size = metadata['batch_size']
        classname = metadata['type']

        from .nets import SingleLayerNeuralNet
        subclass = get_subclass_by_name(cls, classname)

        # TODO (07.08.17): This should be done differently
        for key in subclass.init_args:
            value = subclass.init_args[key]

            if value in metadata:
                kwargs[key] = metadata[value]
            else:
                logger.warning('Trying to fetch non-existing pair (%s, %s) from tw file %s' % (key, value, filename))

        with TFSession(sess) as sess:
            net = subclass(X_shape, y_shape, name=name, sess=sess, **kwargs)
            net.load(weights_filename, sess=sess)
            net.batch_size = batch_size

            return net

    def reset(self, **kwargs):
        """ Resets the value of all variables stored on the graph """

        with TFSession(graph=self._graph, variables=self._variables) as sess:
            sess.run(tf.global_variables_initializer())

    def save(self, path: str, *, sess: tf.Session = None, **kwargs):
        """ Saves the graph to a file using tf.train.Saver. Also stores a more verbose *.tw file
        which will also write additional info given in **kwargs (e.g. accuracy or labels) """

        with TFSession(sess, self.graph, variables=self.variables) as sess:
            saver = tf.train.Saver(tf.trainable_variables())
            saver.save(sess, path, meta_graph_suffix=META_GRAPH_SUFFIX)

            metadata = kwargs
            metadata['name'] = self.name
            metadata['X_shape'] = self.X_shape
            metadata['y_shape'] = self.y_shape
            metadata['time'] = str(datetime.datetime.now())
            metadata['type'] = self.__class__.__name__

            metadata_filename = '%s.%s' % (path, METADATA_SUFFIX)
            with open(metadata_filename, 'w') as f:
                f.write(json.dumps(metadata, indent=2))

    def load(self, path: str, sess: tf.Session = None, **kwargs):
        with TFSession(sess, self.graph) as sess:
            graph_path = path + '.meta'
            saver = tf.train.Saver(tf.trainable_variables())
            saver.restore(sess, path)

            self._graph = sess.graph
            self.X = sess.graph.get_tensor_by_name(self.name + '/X_placeholder:0')
            self.y = sess.graph.get_tensor_by_name(self.name + '/y_placeholder:0')
            self.lr = sess.graph.get_tensor_by_name(self.name + '/learning_rate_placeholder:0')
            self.preds = sess.graph.get_tensor_by_name(self.name + '/pred:0')
            self.loss = sess.graph.get_tensor_by_name(self.name + '/loss:0')
            self.accuracy = sess.graph.get_tensor_by_name(self.name + '/accuracy:0')

            self.checkpoint_variables(sess)


class Predictive(ABC):
    @abstractmethod
    def validate(self, X: np.ndarray, y: np.ndarray, *, sess: tf.Session = None, **kwargs):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, *, sess: tf.Session = None, **kwargs) -> np.ndarray:
        pass


class FixedRegressionModel(Predictive):
    @abstractmethod
    def validate(self, X: np.ndarray, y: np.ndarray, *, sess: tf.Session = None, **kwargs) -> float:
        pass


class FixedClassificationModel(BaseModel, Predictive):
    DEFAULT_LOSS = MeanSoftmaxCrossEntropy
    DEFAULT_ACCURACY = CorrectPred

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        if type(value) is tf.Tensor:
            self._loss = value
        elif isinstance(value, Loss):
            self._loss = value(y=self.y, preds=self.preds, name=self.name + '/loss')
        elif isinstance(value, ABCMeta):
            self._loss = value()(y=self.y, preds=self.preds, name=self.name + '/loss')
        elif callable(value):
            self._loss = value(y=self.y, preds=self.preds, name=self.name + '/loss')
        elif type(value) is str:
            f = Loss.from_name(value)
            self._loss = f(y=self.y, preds=self.preds, name=self.name + '/loss')
        else:
            log_and_raise(InvalidArgumentException, 'Invalid loss type %s. (Valid is [tf.Tensor, Loss, ABCMeta, callable, str])' % str(type(value)))

    @property
    def accuracy(self):
        return self._accuracy

    @accuracy.setter
    def accuracy(self, value):
        if type(value) is tf.Tensor:
            self._accuracy = value
        elif isinstance(value, Accuracy):
            self._accuracy = value(y=self.y, preds=self.preds, name=self.name + '/accuracy')
        elif isinstance(value, ABCMeta):
            self._accuracy = value()(y=self.y, preds=self.preds, name=self.name + '/accuracy')
        elif callable(value):
            self._accuracy = value(y=self.y, preds=self.preds, name=self.name + '/accuracy')
        elif type(value) is str:
            f = get_accuracy_by_name(value)
            self._accuracy = f(y=self.y, preds=self.preds, name=self.name + '/accuracy')
        else:
            log_and_raise(InvalidArgumentException, 'Invalid accuracy type %s. (Valid is [tf.Tensor, Accuracy, ABCMeta, callable, str])' % str(type(value)))
    
    def __init__(self, X_shape: List[int], y_size: Union[int, List[int]], layers: List[Union[tf.Tensor, _tensor_wrapper]], *, preprocessing: List[Union[tf.Tensor, _tensor_wrapper]] = None, sess: tf.Session = None, name: str = 'BaseModel'):
        with TFSession(sess) as sess:
            BaseModel.__init__(self, X_shape, y_size, layers, preprocessing=preprocessing, sess=sess, name=name)

            self.loss = self.DEFAULT_LOSS
            self.accuracy = self.DEFAULT_ACCURACY

    @abstractmethod
    def validate(self, X: np.ndarray, y: np.ndarray, *, sess: tf.Session = None, **kwargs) -> (float, float):
        pass


class Trainable(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, *, epochs: int, sess: tf.Session = None, **kwargs):
        pass


class RegressionModel(FixedRegressionModel, Trainable):
    def hei():
        print('hå')


class ClassificationModel(FixedClassificationModel, Trainable):
    DEFAULT_OPTIMIZER = Adam
    DEFAULT_LEARNING_RATE = 0.01

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer_key = value
        if type(value) is tf.Tensor:
            self._optimizer = value
        elif isinstance(value, Optimizer):
            self._optimizer = value(learning_rate=self.lr_placeholder, loss=self.loss, name=self.name + '/optimizer')
        elif isinstance(value, ABCMeta):
            self._optimizer = value()(learning_rate=self.lr_placeholder, loss=self.loss, name=self.name + '/optimizer')
        elif callable(value):
            self._optimizer = value(learning_rate=self.lr_placeholder, loss=self.loss, name=self.name + '/optimizer')
        elif type(value) is str:
            f = get_optimizer_by_name(value)
            self._optimizer = f(learning_rate=self.lr_placeholder, loss=self.loss, name=self.name + '/optimizer')
        else:
            log_and_raise(InvalidArgumentException, 'Invalid optimizer type %s. (Valid is [tf.Tensor, Optimizer, ABCMeta, callable, str])' % str(type(value)))

    @property
    def loss(self):
        return FixedClassificationModel.loss.fget(self)
    
    @loss.setter
    def loss(self, value):
        FixedClassificationModel.loss.fset(self, value)
        self.optimizer = self._optimizer_key

    def __init__(self, X_shape: List[int], y_size: Union[int, List[int]], layers: List[Union[tf.Tensor, _tensor_wrapper]], *, preprocessing: List[Union[tf.Tensor, _tensor_wrapper]] = None, sess: tf.Session = None, name: str = 'ClassificationModel'):
        with TFSession(sess) as sess:
            self._optimizer_key = self.DEFAULT_OPTIMIZER
            self.lr_placeholder = tf.placeholder_with_default(self.DEFAULT_LEARNING_RATE, [], name=name + '/lr_placeholder')
            self.learning_rate = self.DEFAULT_LEARNING_RATE

            FixedClassificationModel.__init__(self, X_shape, y_size, layers, preprocessing=preprocessing, sess=sess, name=name)

    def save(self, *args, **kwargs):
        super().save(*args, batch_size=self.batch_size, **kwargs)


class Derivable(ABC):
    @abstractmethod
    def extract_features(self, layer: str = None, *, sess: tf.Session = None, **kwargs):
        pass

