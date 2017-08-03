import datetime

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod, abstractproperty
from typing import Callable, Dict, List, Union

from tfwrapper import TFSession
from tfwrapper.layers import Layer
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
    
    def __init__(self, X_shape: list, y_size: Union[int, list], layers: List[Union[tf.Tensor, tensor_wrapper]], preprocessing=None, sess=None, name='BaseModel'):
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
            self.tensors, self.pred = _parse_layer_list(self.X, self.layers)
            self.loss = self.loss_function() # THIS SHOULD NOT BE HERE
            self.optimizer = self.optimizer_function() # THIS SHOULD NOT BE HERE
            self.accuracy = self.accuracy_function() # THIS SHOULD NOT BE HERE
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

    @abstractmethod
    def accuracy_function(self) -> tf.Tensor:
        pass

    @abstractmethod
    def loss_function(self) -> tf.Tensor:
        pass

class Predictive(ABC):
    @abstractmethod
    def validate(self, X: np.ndarray, y: np.ndarray, *, sess: tf.Session = None, **kwargs):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, *, sess: tf.Session = None, **kwargs) -> np.ndarray:
        pass


class RegressionModel(Predictive):
    @abstractmethod
    def validate(self, X: np.ndarray, y: np.ndarray, *, sess: tf.Session = None, **kwargs) -> float:
        pass


class ClassificationModel(Predictive):
    @abstractmethod
    def validate(self, X: np.ndarray, y: np.ndarray, *, sess: tf.Session = None, **kwargs) -> (float, float):
        pass


class Trainable(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, *, epochs: int, sess: tf.Session = None, **kwargs):
        pass


class Derivable(ABC):
    @abstractmethod
    def extract_features(self, layer: str = None, *, sess: tf.Session = None, **kwargs):
        pass

