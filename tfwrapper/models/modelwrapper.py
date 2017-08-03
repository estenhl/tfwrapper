from abc import ABC
import numpy as np
import tensorflow as tf

from tfwrapper.dataset import Dataset, ImageDataset, ImagePreprocessor
from tfwrapper.models import BaseModel, RegressionModel, ClassificationModel
from tfwrapper.utils.exceptions import log_and_raise, InvalidArgumentException

from .basemodel import RegressionModel, ClassificationModel
from .metamodel import RegressionMetaModel, ClassificationMetaModel

class ModelWrapper(ABC):
    """ A class which packs regular models with the MetaModel interface """

    @staticmethod
    def from_instance(obj):
        if isinstance(obj, RegressionModel):
            return RegressionModelWrapper(obj)
        elif isinstance(obj, ClassificationModel):
            return ClassificationModelWrapper(obj)
        else:
            log_and_raise(InvalidArgumentException, 'Class %s does not have a Wrapper! (Valid is [RegressionModel, ClassificationModel]' % obj.__class__.__name__)

    @property
    def graph(self):
        return self.model.graph

    @property
    def variables(self):
        return self.model.variables

    def __init__(self, model: BaseModel):
        self.model = model

    def train(self, dataset: Dataset, *, epochs: int, preprocessor: ImagePreprocessor = None, sess: tf.Session = None, **kwargs):
        if isinstance(dataset, ImageDataset) and preprocessor is not None:
            dataset.loader.preprocessor = preprocessor

        return self.model.train(dataset.X, dataset.y, epochs=epochs, sess=sess, **kwargs)

    def validate(self, dataset: Dataset, *, preprocessor: ImagePreprocessor = None, sess: tf.Session = None, **kwargs):
        if isinstance(dataset, ImageDataset) and preprocessor is not None:
            dataset.loader.preprocessor = preprocessor

        return self.model.validate(dataset.X, dataset.y, sess=sess)

    def predict(self, dataset: Dataset, *, preprocessor: ImagePreprocessor = None, sess: tf.Session = None, **kwargs) -> np.ndarray:
        if isinstance(dataset, ImageDataset) and preprocessor is not None:
            dataset.loader.preprocessor = preprocessor

        return self.model.predict(dataset.X, dataset.y, sess=sess)

    def reset(self, **kwargs):
        return self.model.reset()

    def save(self, path: str, sess: tf.Session = None, **kwargs):
        self.model.save(path, sess=sess)

    def from_tw(self, path, sess=None, **kwargs):
        from tfwrapper.models.nets import NeuralNet
        self.model = NeuralNet.from_tw(path, sess=sess)


class RegressionModelWrapper(ModelWrapper, RegressionMetaModel):
    def __init__(self, model: RegressionModel, name: str = 'RegressionModelWrapper'):
        ModelWrapper.__init__(self, model)
        RegressionMetaModel.__init__(self, name)

    def validate(self, dataset: Dataset, *, preprocessor: ImagePreprocessor = None, sess: tf.Session = None, **kwargs) -> float:
        return ModelWrapper.validate(self, dataset, preprocessor=preprocessor, sess=sess, **kwargs)


class ClassificationModelWrapper(ModelWrapper, ClassificationMetaModel):
    def __init__(self, model: ClassificationModel, name: str = 'ClassificationModelWrapper'):
        ModelWrapper.__init__(self, model)
        RegressionMetaModel.__init__(self, name)

    def validate(self, dataset: Dataset, *, preprocessor: ImagePreprocessor = None, sess: tf.Session = None, **kwargs) -> (float, float):
        return ModelWrapper.validate(self, dataset, preprocessor=preprocessor, sess=sess, **kwargs)
