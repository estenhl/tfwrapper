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

    def _compute_predictions(self, dataset, sess=None):
        preds = self.prediction_models[0].predict(dataset, sess=sess)
        combined_preds = np.expand_dims(preds, axis=0)

        for model in self.prediction_models[1:]:
            preds = model.predict(dataset, sess=sess)
            preds = np.expand_dims(preds, axis=0)
            combined_preds = np.concatenate([combined_preds, preds], axis=0)
        
        combined_preds = np.argmax(combined_preds, axis=2)
        combined_preds = np.transpose(combined_preds)

        return combined_preds

    def train(self, dataset, *, epochs, sess=None):
        for model in self.prediction_models:
            print('Starting training model %s %s' % (model.name, time.time()))
            model.train(dataset, epochs=epochs, sess=sess)
            print('Finished training model %s %s' % (model.name, time.time()))

        preds = self._compute_predictions(dataset, sess=sess)

        with TFSession(sess, self.decision_model.graph) as sess:
            self.decision_model.train(preds, dataset.y, epochs=epochs, sess=sess)

    def validate(self, dataset, sess=None):
        preds = self._compute_predictions(dataset, sess=sess)

        variables = self.decision_model.variables
        with TFSession(sess, self.decision_model.graph, variables=variables) as sess:
            return self.decision_model.validate(preds, dataset.y, sess=sess)