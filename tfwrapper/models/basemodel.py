import datetime

import numpy as np
import tensorflow as tf
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from typing import Callable, Dict, List, Union

from tfwrapper import TFSession
from tfwrapper.layers import Layer
from tfwrapper.layers.accuracy import Accuracy, CorrectPred
from tfwrapper.layers.loss import Loss, MeanSoftmaxCrossEntropy
from tfwrapper.layers.optimizers import Optimizer, Adam
from tfwrapper.utils.exceptions import log_and_raise, InvalidArgumentException


tensor_wrapper = Callable[[Union[tf.Tensor, List[tf.Tensor]]], tf.Tensor]


def _parse_layer_list(start: tf.Tensor, remaining: List[Union[tf.Tensor, tensor_wrapper]], sess: tf.Session = None):
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
    def graph(self) -> tf.Graph:
        return self._graph

    def variables(self) -> Dict[str, tf.Tensor]:
        return self._variables
    
    def __init__(self, X_shape: List[int], y_size: Union[int, List[int]], layers: List[Union[tf.Tensor, tensor_wrapper]], *, preprocessing: List[Union[tf.Tensor, tensor_wrapper]] = None, sess: tf.Session = None, name: str = 'BaseModel'):
        if preprocessing is None:
            preprocessing = []

        self.name = name
        self.X_shape = X_shape
        if type(y_size) is int:
            self.y_size = [y_size]
        elif type(y_size) is list:
            self.y_size = y_size
        else:
            log_and_raise(InvalidArgumentException, 'Invalid type %s for y_size. (Valid is [int, list])' % str(type(y_size)))

        with TFSession(sess) as sess:
            self.X = tf.placeholder(tf.float32, [None] + self.X_shape, name=self.name + '/X_placeholder')
            self.y = tf.placeholder(tf.float32, [None] + self.y_size, name=self.name + '/y_placeholder')
            self.lr = tf.placeholder(tf.float32, [], name=self.name + '/learning_rate_placeholder')

            self.layers = preprocessing + layers
            self.tensors, self.preds = _parse_layer_list(self.X, self.layers)
            self._graph = sess.graph

        self._variables = {}
        self.init_vars_when_training = True
        self.feed_dict = {}

    @abstractmethod
    def reset(self, **kwargs):
        pass

    @abstractmethod
    def save(self, path: str, *, sess: tf.Session = None, **kwargs):
        pass

    @abstractmethod
    def load(self, path: str, **kwargs):
        pass

    @abstractmethod
    def from_tw(self, path: str, sess: tf.Session = None, **kwargs):
        pass


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
    
    def __init__(self, X_shape: List[int], y_size: Union[int, List[int]], layers: List[Union[tf.Tensor, tensor_wrapper]], *, preprocessing: List[Union[tf.Tensor, tensor_wrapper]] = None, sess: tf.Session = None, name: str = 'BaseModel'):
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

    def __init__(self, X_shape: List[int], y_size: Union[int, List[int]], layers: List[Union[tf.Tensor, tensor_wrapper]], *, preprocessing: List[Union[tf.Tensor, tensor_wrapper]] = None, sess: tf.Session = None, name: str = 'ClassificationModel'):
        with TFSession(sess) as sess:
            self._optimizer_key = self.DEFAULT_OPTIMIZER
            self.lr_placeholder = tf.placeholder_with_default(self.DEFAULT_LEARNING_RATE, [], name=name + '/lr_placeholder')
            self.learning_rate = self.DEFAULT_LEARNING_RATE

            FixedClassificationModel.__init__(self, X_shape, y_size, layers, preprocessing=preprocessing, sess=sess, name=name)


class Derivable(ABC):
    @abstractmethod
    def extract_features(self, layer: str = None, *, sess: tf.Session = None, **kwargs):
        pass

