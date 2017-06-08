import numpy as np

from tfwrapper import logger
from tfwrapper import TFSession
from tfwrapper.dataset import Dataset

import time


class Stacker():
    def __init__(self, prediction_models, decision_model, name='Stacker'):
        self.prediction_models = prediction_models
        self.decision_model = decision_model
        self.name = name

    def _compute_predictions(self, dataset, preprocessor=None, sess=None):
        preds = self.prediction_models[0].predict(dataset, sess=sess)
        combined_preds = np.expand_dims(preds, axis=0)

        for model in self.prediction_models[1:]:
            preds = model.predict(dataset, preprocessor=preprocessor, sess=sess)
            preds = np.expand_dims(preds, axis=0)
            combined_preds = np.concatenate([combined_preds, preds], axis=0)
        
        n_models, n_samples, n_features = combined_preds.shape
        p2 = np.zeros((n_samples, n_models, n_features))
        for i in range(n_models):
            for j in range(n_samples):
                for k in range(n_features):
                    p2[j][i][k] = combined_preds[i][j][k]

        return p2

    def train(self, dataset, *, epochs, preprocessor=None, sess=None):
        if type(epochs) is int:
            epochs = [epochs, epochs]
        elif type(epochs) is list:
            pass
        else:
            # TODO (08.06.17): Fix exception and stuff
            logger.error('Invalid epochs')
        for model in self.prediction_models:
            model.train(dataset, preprocessor=preprocessor, epochs=epochs[0], sess=sess)

        preds = self._compute_predictions(dataset, preprocessor=preprocessor, sess=sess)

        with TFSession(sess, self.decision_model.graph) as sess:
            self.decision_model.train(preds, dataset.y, epochs=epochs[1], sess=sess)

    def validate(self, dataset, preprocessor=None, sess=None):
        preds = self._compute_predictions(dataset, preprocessor=preprocessor, sess=sess)

        for model in self.prediction_models:
            _, acc = model.validate(dataset, preprocessor=preprocessor, sess=sess)
            print('%s acc: %.2f%%' % (model.name, (acc * 100)))

        variables = self.decision_model.variables
        with TFSession(sess, self.decision_model.graph, variables=variables) as sess:
            return self.decision_model.validate(preds, dataset.y, sess=sess)