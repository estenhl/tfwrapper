import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod, abstractproperty


class BaseModel(ABC):
    @abstractproperty
    def graph(self):
        pass

    @abstractproperty
    def variables(self):
        pass
    
    def __init__(self, name):
        self.name = name

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

