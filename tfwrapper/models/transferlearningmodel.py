import json
import datetime
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper import Dataset
from tfwrapper import FeatureLoader
from tfwrapper import ImagePreprocessor
from tfwrapper import METADATA_SUFFIX
from tfwrapper.models import Predictive
from tfwrapper.models import PredictiveRegressionModel
from tfwrapper.models import PredictiveClassificationModel
from tfwrapper.utils.data import get_subclass_by_name

from .frozenmodel import FrozenModel
from .metamodel import MetaModel, RegressionMetaModel, ClassificationMetaModel


class TransferLearningModel(MetaModel):
    def __init__(self, feature_model: Predictive, prediction_model: Predictive, features_layer: str = None, features_cache: str = None, name: str = 'TransferLearningModel'):
        super().__init__(name)

        self.feature_model = feature_model
        self.prediction_model = prediction_model
        self.features_layer = features_layer
        self.features_cache = features_cache

        if self.features_layer is None:
            self.features_layer = feature_model.bottleneck_tensor

    def train(self, dataset: Dataset, *, epochs: int, preprocessor: ImagePreprocessor = None, sess: tf.Session = None):
        with TFSession(sess, self.feature_model.graph) as sess1:
            dataset.loader = FeatureLoader(self.feature_model, cache=self.features_cache, preprocessor=preprocessor, sess=sess1)
            X, y = dataset.X, dataset.y

        with TFSession(sess, self.prediction_model.graph) as sess2:
            self.prediction_model.train(X, y, epochs=epochs, sess=sess2)

    def validate(self, dataset: Dataset, *, preprocessor: ImagePreprocessor = None, sess: tf.Session = None):
        with TFSession(sess, self.feature_model.graph) as sess1:
            dataset.loader = FeatureLoader(self.feature_model, cache=self.features_cache, preprocessor=preprocessor, sess=sess1)
            X, y = dataset.X, dataset.y

        variables = self.prediction_model.variables
        with TFSession(sess, self.prediction_model.graph, variables=variables) as sess2:
            return self.prediction_model.validate(X, y, sess=sess2)


    def predict(self, dataset: Dataset, *, preprocessor: ImagePreprocessor = None, sess: tf.Session = None):
        with TFSession(sess, self.feature_model.graph) as sess1:
            dataset.loader = FeatureLoader(self.feature_model, cache=self.features_cache, preprocessor=preprocessor, sess=sess1)
            X = dataset.X

        variables = self.prediction_model.variables
        with TFSession(sess, self.prediction_model.graph, variables=variables) as sess2:
            return self.prediction_model.predict(X, sess=sess2)

    def reset(self):
        self.prediction_model.reset()

    def save(self, path: str, *, sess: tf.Session = None, **kwargs):
        prediction_model_path = path + '_prediction'
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

        self.prediction_model.save(prediction_model_path, sess=sess)

    @staticmethod
    def from_tw(path: str, sess: tf.Session = None, **kwargs):
        metadata_filename = '%s.%s' % (path, METADATA_SUFFIX)
        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)

        name = metadata['name']
        feature_model_type = metadata['feature_model_type']
        prediction_model_type = metadata['prediction_model_type']
        prediction_model_path = metadata['prediction_model_path']
        features_layer = metadata['features_layer']
        features_cache = metadata['features_cache']

        feature_model = FrozenModel.from_type(feature_model_type)
        prediction_model = SupervisedModel.from_tw(prediction_model_path, sess=sess)

        return TransferLearningModel(feature_model, prediction_model, features_layer=features_layer, features_cache=features_cache, name=name)

class TransferLearningClassificationModel(TransferLearningModel, ClassificationMetaModel):
    def __init__(self, feature_model: Predictive, prediction_model: PredictiveClassificationModel, features_layer: str = None, features_cache: str = None, name: str = 'TransferLearningModel'):
        TransferLearningModel.__init__(self, feature_model, prediction_model, features_layer, feature_cache, name)

class TransferLearningRegressionModel(TransferLearningModel, RegressionMetaModel):
    def __init__(self, feature_model: Predictive, prediction_model: PredictiveRegressionModel, features_layer: str = None, features_cache: str = None, name: str = 'TransferLearningModel'):
        TransferLearningModel.__init__(self, feature_model, prediction_model, features_layer, feature_cache, name)


