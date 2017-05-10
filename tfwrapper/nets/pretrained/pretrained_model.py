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

    def get_layer_shape(self, id):
        if hasattr(self, 'core_layer_names'):
            if type(id) is int:
                return self.core_layer_names[id]['output']
            elif type(id) is str:
                for i in range(len(self.core_layer_names)):
                    if self.core_layer_names[i]['name'] == id:
                        return self.core_layer_names[i]['output']

            logger.error('Invalid id %s' % repr(id))
        return [-1, -1, -1, -1]

