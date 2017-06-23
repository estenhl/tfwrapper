from abc import ABC
from abc import abstractmethod

class MetaModel(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def train(self, dataset, *, epochs, preprocessor=None, sess=None, **kwargs):
        pass

    @abstractmethod
    def validate(self, dataset, *, preprocessor=None, sess=None, **kwargs):
        pass

    @abstractmethod
    def predict(self, dataset, *, preprocessor=None, sess=None, **kwargs):
        pass

    @abstractmethod
    def reset(self, **kwargs):
        pass

    @abstractmethod
    def save(self, path, sess=None, **kwargs):
        pass

    @abstractmethod
    def from_tw(path, sess=None, **kwargs):
        pass