import functools
import numpy as np

from tfwrapper import logger
from tfwrapper import TFSession
from tfwrapper.dataset import Dataset
from tfwrapper.utils.exceptions import raise_exception
from tfwrapper.utils.exceptions import InvalidArgumentException

import time


class Stacker():
    DATA_POLICY_FOLDS = 'folds'
    DATA_POLICY_SHUFFLED_FOLDS = 'shuffle'
    _VALID_DATA_POLICIES = [None, DATA_POLICY_FOLDS, DATA_POLICY_SHUFFLED_FOLDS]

    def __init__(self, prediction_models, decision_model, policy=None, name='Stacker'):
        self.prediction_models = prediction_models
        self.decision_model = decision_model
        self.name = name

        if not policy in self._VALID_DATA_POLICIES:
            raise_exception('Invalid data policy %s. (Valid is %s)' % (policy, str(self._VALID_DATA_POLICIES)), InvalidArgumentException)
        
        self._policy = policy

    def _reorder_predictions(self, predictions):
        n_models, n_samples, n_features = predictions.shape
        reordered = np.zeros((n_samples, n_models, n_features))
        for i in range(n_models):
            for j in range(n_samples):
                reordered[j][i] = predictions[i][j]

        return reordered

    def _compute_predictions(self, dataset, preprocessor=None, sess=None):
        preds = self.prediction_models[0].predict(dataset, preprocessor=preprocessor, sess=sess)
        combined_preds = np.expand_dims(preds, axis=0)

        for i in range(1, len(self.prediction_models)):
            model = self.prediction_models[i]
            preds = model.predict(dataset, preprocessor=preprocessor, sess=sess)
            preds = np.expand_dims(preds, axis=0)
            combined_preds = np.concatenate([combined_preds, preds], axis=0)
        
        reordered = self._reorder_predictions(combined_preds)

        return reordered

    def _train_on_dataset(self, dataset, *, epochs, preprocessor=None, sess=None):
        for model in self.prediction_models:
            model.train(dataset, preprocessor=preprocessor, epochs=epochs[0], sess=sess)

        preds = self._compute_predictions(dataset, preprocessor=preprocessor, sess=sess)

        with TFSession(sess, self.decision_model.graph) as sess:
            self.decision_model.train(preds, dataset.y, epochs=epochs[1], sess=sess)

    def _train_on_folds(self, dataset, *, epochs, preprocessor=None, sess=None):
        num_models = len(self.prediction_models)
        folds = dataset.folds(num_models)
        num_folds = num_models

        # Create training and validation sets
        training_sets = []
        prediction_sets = []
        for i in range(num_folds):
            training_sets.append(functools.reduce(lambda x, y: x + y, folds[:i] + folds[i + 1:],))
            prediction_sets.append(folds[i])

        # Create predictions by training the individual models on folds
        preds = []
        for i in range(num_models):
            model = self.prediction_models[i]
            for j in range(num_folds):
                model.train(training_sets[j], epochs=epochs[0], preprocessor=preprocessor, sess=sess)
                if j == 0:
                    preds.append(model.predict(prediction_sets[j], preprocessor=preprocessor, sess=sess))
                else:
                    preds[i] = np.concatenate([preds[i], model.predict(prediction_sets[j], preprocessor=preprocessor, sess=sess)], axis=0)
                model.reset()
        preds = np.asarray(preds)
        preds = self._reorder_predictions(preds)

        # Train the base models
        for model in self.prediction_models:
            model.train(dataset, epochs=epochs[0], preprocessor=preprocessor, sess=sess)

        # Train the decision model
        with TFSession(sess, self.decision_model.graph) as sess:
            self.decision_model.train(preds, dataset.y, epochs=epochs[1], sess=sess)

    def _train_on_shuffled_folds(self, dataset, *, epochs, preprocessor=None, sess=None):
        print('Training on shuffled folds')

    def train(self, dataset, *, epochs, preprocessor=None, sess=None):
        if type(epochs) is int:
            epochs = [epochs, epochs]
        elif type(epochs) is list:
            pass
        else:
            raise_exception('Invalid epochs type %s. (Valid is [int, list])' % repr(type(epochs)), InvalidArgumentException)

        if self._policy is None:
            self._train_on_dataset(dataset, epochs=epochs, preprocessor=preprocessor, sess=sess)
        elif self._policy is self.DATA_POLICY_FOLDS:
            self._train_on_folds(dataset, epochs=epochs, preprocessor=preprocessor, sess=sess)
        elif self._policy is self.DATA_POLICY_SHUFFLED_FOLDS:
            self._train_on_shuffled_folds(dataset, epochs=epochs, preprocessor=preprocessor, sess=sess)
        else:
            raise_exception('Invalid data policy %s. (Valid is %s)' % (policy, str(self._VALID_DATA_POLICIES)), InvalidArgumentException)
        
    def validate(self, dataset, preprocessor=None, sess=None):
        preds = self._compute_predictions(dataset, preprocessor=preprocessor, sess=sess)

        for model in self.prediction_models:
            _, acc = model.validate(dataset, preprocessor=preprocessor, sess=sess)
            print('%s acc: %.2f%%' % (model.name, (acc * 100)))

        variables = self.decision_model.variables
        with TFSession(sess, self.decision_model.graph, variables=variables) as sess:
            return self.decision_model.validate(preds, dataset.y, sess=sess)

    def predict(self, dataset, preprocessor=None, sess=None):
        preds = self._compute_predictions(dataset, preprocessor=preprocessor, sess=sess)

        variables = self.decision_model.variables
        with TFSession(sess, self.decision_model.graph, variables=variables) as sess:
            return self.decision_model.predict(preds, dataset.y, sess=sess)