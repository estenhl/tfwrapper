from tfwrapper import TFSession
from tfwrapper import FeatureLoader
from tfwrapper import ImagePreprocessor

class TransferLearningModel():
    def __init__(self, feature_model, prediction_model, features_layer=None, features_cache=None, name='TransferLearningModel'):
        self.feature_model = feature_model
        self.prediction_model = prediction_model
        self.features_layer = features_layer
        if self.features_layer is None:
            self.features_layer = feature_model.bottleneck_tensor
        self.features_cache = features_cache
        self.name = name

    def train(self, dataset, *, epochs, preprocessor=None, sess=None):
        with TFSession(sess, self.feature_model.graph) as sess1:
            if preprocessor is None:
                preprocessor = ImagePreprocessor()

            dataset.loader = FeatureLoader(self.feature_model, cache=self.features_cache, preprocessor=preprocessor, sess=sess1)
            X, y = dataset.X, dataset.y

        with TFSession(sess, self.prediction_model.graph) as sess2:
            self.prediction_model.train(X, y, epochs=epochs, sess=sess2)

    def validate(self, dataset, *, preprocessor=None, sess=None):
        with TFSession(sess, self.feature_model.graph) as sess1:
            if preprocessor is None:
                preprocessor = ImagePreprocessor()

            dataset.loader = FeatureLoader(self.feature_model, cache=self.features_cache, preprocessor=preprocessor, sess=sess1)
            X, y = dataset.X, dataset.y

        with TFSession(sess, self.prediction_model.graph) as sess2:
            return self.prediction_model.validate(X, y, sess=sess2)

    def predict(self, dataset, *, preprocessor=None, sess=None):
        with TFSession(sess, self.feature_model.graph) as sess1:
            if preprocessor is None:
                preprocessor = ImagePreprocessor()

            dataset.loader = FeatureLoader(self.feature_model, cache=self.features_cache, preprocessor=preprocessor, sess=sess1)
            X = dataset.X

        with TFSession(sess, self.prediction_model.graph) as sess2:
            return self.prediction_model.predict(X, sess=sess2)