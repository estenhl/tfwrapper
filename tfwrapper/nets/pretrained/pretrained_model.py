from abc import ABC, abstractmethod

from tfwrapper import TFSession

class PretrainedModel():
    def __init__(self, graph):
        self.graph = graph

    @abstractmethod
    def run_op(self, to_layer, from_layer, data, sess):
        raise NotImplementedException('PretrainedModel is an abstract class')

    @abstractmethod
    def extract_feature(self, img, sess, layer):
        raise NotImplementedException('PretrainedModel is an abstract class')

    @abstractmethod
    def extract_features_from_file(self, filename, layer, sess):
        raise NotImplementedException('PretrainedModel is an abstract class')

    def extract_bottleneck_features(self, img, sess=None):
        with TFSession(sess, self.graph) as sess:
            features = self.run_op(self.DEFAULT_FEATURES_LAYER, self.DEFAULT_INPUT_LAYER, img, sess)
            features = features.flatten()

            return features

