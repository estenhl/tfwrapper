import numpy as np

class TransferLearningModel():
    def __init__(self, features_model, features_layer, features_shape, prediction_model):
        self.features_model = features_model
        self.features_layer = features_layer
        self.features_shape = features_shape
        self.prediction_model = prediction_model

    def train(self, X, y, *, epochs, sess):
        features = []

        for x in X:
            features.append(self.features_model.get_feature(x, sess=sess, layer=self.features_layer))

        features = np.asarray(features)
        features = np.reshape(features, [-1] + self.features_shape)

        self.prediction_model.train(features, y, sess=sess, epochs=epochs)