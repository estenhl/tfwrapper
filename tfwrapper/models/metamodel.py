import numpy as np
import tensorflow as tf
from abc import ABC
from abc import abstractmethod

from tfwrapper import Dataset
from tfwrapper import ImagePreprocessor

class MetaModel(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def train(self, dataset: Dataset, *, epochs: int, preprocessor: ImagePreprocessor = None, sess: tf.Session = None, **kwargs):
        pass

    @abstractmethod
    def reset(self, **kwargs):
        pass

    @abstractmethod
    def save(self, path, sess=None, **kwargs):
        pass

    @abstractmethod
    def from_tw(self, path, sess=None, **kwargs):
        pass


class PredictiveMeta(ABC):
    @abstractmethod
    def validate(self, dataset: Dataset, *, preprocessor: ImagePreprocessor = None, sess: tf.Session = None, **kwargs):
        pass

    @abstractmethod
    def predict(self, dataset: Dataset, *, preprocessor: ImagePreprocessor = None, sess: tf.Session = None, **kwargs) -> np.ndarray:
        pass



class RegressionMetaModel(MetaModel, PredictiveMeta):
    @abstractmethod
    def validate(self, dataset: Dataset, *, preprocessor: ImagePreprocessor = None, sess: tf.Session = None, **kwargs) -> float:
        pass


class ClassificationMetaModel(MetaModel, PredictiveMeta):
    @abstractmethod
    def validate(self, dataset: Dataset, *, preprocessor: ImagePreprocessor = None, sess: tf.Session = None, **kwargs) -> (float, float):
        pass