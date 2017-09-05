import json
import datetime
import numpy as np
import os
import tensorflow as tf
from typing import Union

from tfwrapper import Dataset, FeatureLoader, ImageDataset, ImagePreprocessor, logger, METADATA_SUFFIX, TFSession
from tfwrapper.models.nets import NeuralNet
from tfwrapper.utils.data import get_subclass_by_name
from tfwrapper.utils.exceptions import log_and_raise, InvalidArgumentException

from .basemodel import Predictive, Derivable, RegressionModel, ClassificationModel
from .frozenmodel import FrozenModel
from .metamodel import MetaModel, PredictiveMeta, RegressionMetaModel, ClassificationMetaModel
from .modelwrapper import ModelWrapper


class TransferLearningModel(MetaModel, PredictiveMeta):
    def __init__(self, feature_model: Derivable, prediction_model: Union[Predictive, PredictiveMeta], features_layer: str = None, features_cache: str = None, name: str = 'TransferLearningModel'):
        super().__init__(name)

        if isinstance(feature_model, Derivable):
            self.feature_model = feature_model
        else:
            log_and_raise(InvalidArgumentException, 'feature_model must be a Derivable model')

        if isinstance(prediction_model, Predictive):
            self.prediction_model = ModelWrapper(prediction_model)
        elif isinstance(prediction_model, PredictiveMeta):
            self.prediction_model = prediction_model
        else:
            log_and_raise(InvalidArgumentException, 'prediction_model must either be a Predictive model, or a PredictiveMeta model')

        self.features_layer = features_layer
        self.features_cache = features_cache

        if self.features_layer is None:
            self.features_layer = feature_model.bottleneck_tensor

    def train(self, dataset: Dataset, *, epochs: int, preprocessor: ImagePreprocessor = None, sess: tf.Session = None, **kwargs):
        with TFSession(sess, self.feature_model.graph) as sess1:
            dataset.loader = FeatureLoader(self.feature_model, cache=self.features_cache, preprocessor=preprocessor, sess=sess1)

        with TFSession(sess, self.prediction_model.graph) as sess2:
            self.prediction_model.train(dataset, epochs=epochs, sess=sess2, **kwargs)

    def validate(self, dataset: Dataset, *, preprocessor: ImagePreprocessor = None, sess: tf.Session = None):
        with TFSession(sess, self.feature_model.graph) as sess1:
            dataset.loader = FeatureLoader(self.feature_model, cache=self.features_cache, preprocessor=preprocessor, sess=sess1)

        variables = self.prediction_model.variables
        with TFSession(sess, self.prediction_model.graph, variables=variables) as sess2:
            return self.prediction_model.validate(dataset, sess=sess2)

    def predict(self, dataset: Dataset = None, *, X = None, preprocessor: ImagePreprocessor = None, sess: tf.Session = None):
        if X is not None and dataset is None:
            dataset = Dataset(X)
        with TFSession(sess, self.feature_model.graph) as sess1:
            if isinstance(dataset, ImageDataset):
                dataset.loader = FeatureLoader(self.feature_model, cache=self.features_cache, preprocessor=preprocessor, sess=sess1)
                X = dataset.X
            else:
                X = self.feature_model.extract_features(dataset.X, self.features_layer, sess=sess)

        dataset = Dataset(X=X, y=np.zeros(len(X)))
        variables = self.prediction_model.variables
        with TFSession(sess, self.prediction_model.graph, variables=variables) as sess2:
            return self.prediction_model.predict(dataset, sess=sess2)

    def reset(self):
        self.prediction_model.reset()

    def save(self, path: str, *, sess: tf.Session = None, **kwargs):
        prediction_model_path = path + '_prediction'

        if not 'labels' in kwargs:
            logger.warning('Saving %s without saving labels. Shame on you' % self.name)

        metadata = kwargs
        metadata['name'] = self.name
        metadata['time'] = str(datetime.datetime.now())
        metadata['feature_model_type'] = self.feature_model.__class__.__name__
        metadata['prediction_model_type'] = self.prediction_model.__class__.__name__
        metadata['prediction_model_path'] = '%s.%s' % (prediction_model_path, 'tw')
        metadata['features_layer'] = self.features_layer
        metadata['features_cache'] = self.features_cache

        metadata_filename = '%s.%s' % (path, METADATA_SUFFIX)
        with open(metadata_filename, 'w') as f:
            f.write(json.dumps(metadata, indent=2))

        if 'labels' in kwargs:
            self.prediction_model.save(prediction_model_path, labels=kwargs['labels'], sess=sess)
        else:
            self.prediction_model.save(prediction_model_path, sess=sess)

    @staticmethod
    def from_tw(path: str, sess: tf.Session = None, **kwargs):
        metadata_filename = '%s.%s' % (path, METADATA_SUFFIX)
        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)

        name = metadata['name']
        feature_model_type = metadata['feature_model_type']
        prediction_model_type = metadata['prediction_model_type']
        prediction_model_path = os.path.join(os.path.split(path)[0], metadata['prediction_model_path'])
        features_layer = metadata['features_layer']
        features_cache = metadata['features_cache']

        feature_model = FrozenModel.from_type(feature_model_type)
        prediction_model = NeuralNet.from_tw(prediction_model_path, sess=sess)

        return TransferLearningModel(feature_model, prediction_model, features_layer=features_layer, features_cache=features_cache, name=name)


class TransferLearningRegressionModel(TransferLearningModel, RegressionMetaModel):
    def __init__(self, feature_model: Predictive, prediction_model: [RegressionModel, RegressionMetaModel], features_layer: str = None, features_cache: str = None, name: str = 'TransferLearningModel'):
        TransferLearningModel.__init__(self, feature_model, prediction_model, features_layer, feature_cache, name)

    def validate(self, dataset: Dataset, *, preprocessor: ImagePreprocessor = None, sess: tf.Session = None) -> float:
        super().validate(dataset, preprocessor=preprocessor, sess=sess)


class TransferLearningClassificationModel(TransferLearningModel, ClassificationMetaModel):
    def __init__(self, feature_model: Predictive, prediction_model: Union[ClassificationModel, ClassificationMetaModel], features_layer: str = None, features_cache: str = None, name: str = 'TransferLearningModel'):
        TransferLearningModel.__init__(self, feature_model, prediction_model, features_layer, feature_cache, name)

    def validate(self, dataset: Dataset, *, preprocessor: ImagePreprocessor = None, sess: tf.Session = None) -> (float, float):
        super().validate(dataset, preprocessor=preprocessor, sess=sess)