import numpy as np
from abc import ABC
from abc import abstractmethod


class BaseModel(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def reset(self, **kwargs):
        pass

    @abstractmethod
    def save(self, path: str, *, sess=None, **kwargs):
        pass

    @abstractmethod
    def load(self, path: str, **kwargs):
        pass

    @abstractmethod
    def from_tw(self, path: str, sess=None, **kwargs):
        pass


class Predictive(ABC):
    @abstractmethod
    def validate(self, X: np.ndarray, y: np.ndarray, *, sess=None, **kwargs):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, *, sess=None, **kwargs) -> np.ndarray:
        pass


class PredictiveRegressionModel(Predictive):
    @abstractmethod
    def validate(self, X: np.ndarray, y: np.ndarray, *, sess=None, **kwargs) -> float:
        pass


class PredictiveClassificationModel(Predictive):
    @abstractmethod
    def validate(self, X: np.ndarray, y: np.ndarray, *, sess=None, **kwargs) -> (float, float):
        pass


class TrainableModel(BaseModel, Predictive):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, *, epochs: int, sess=None, **kwargs):
        pass