import json
import math
import functools
import numpy as np
import tensorflow as tf
from typing import Union, List

from tfwrapper import logger, TFSession, METADATA_SUFFIX
from tfwrapper.dataset import Dataset, ImagePreprocessor
from tfwrapper.models import Predictive, RegressionModel, ClassificationModel, MetaModel, PredictiveMeta, RegressionMetaModel, ClassificationMetaModel, TransferLearningModel, ModelWrapper
from tfwrapper.models.nets import NeuralNet, SingleLayerNeuralNet
from tfwrapper.utils.exceptions import log_and_raise, InvalidArgumentException


class Stacker(MetaModel, PredictiveMeta):
    DATA_POLICY_FOLDS = 'folds'
    DATA_POLICY_SHUFFLED_FOLDS = 'shuffle'
    _VALID_DATA_POLICIES = [None, DATA_POLICY_FOLDS, DATA_POLICY_SHUFFLED_FOLDS]

    def __init__(self, prediction_models: List[Union[Predictive, PredictiveMeta]], decision_model: Union[Predictive, PredictiveMeta], policy: str = None, name: str = 'Stacker'):
        super().__init__(name)

        self.prediction_models = []
        for model in prediction_models:
            if isinstance(model, Predictive):
                self.prediction_models.append(ModelWrapper.from_instance(model))
            elif isinstance(model, PredictiveMeta):
                self.prediction_models.append(model)
            else:
                log_and_raise(InvalidArgumentException, 'Elements of prediction_models must either be a subclass of Predictive or PredictiveMeta')

        if isinstance(decision_model, Predictive):
            self.decision_model = ModelWrapper.from_instance(decision_model)
        elif isinstance(decision_model, PredictiveMeta):
            self.decision_model = decision_model
        else:
            log_and_raise(InvalidArgumentException, 'decision_model must either be a subclass of Predictive or PredictiveMeta')

        if not policy in self._VALID_DATA_POLICIES:
            log_and_raise(InvalidArgumentException, 'Invalid data policy %s. (Valid is %s)' % (policy, str(self._VALID_DATA_POLICIES)))
        
        self._policy = policy

    def _reorder_predictions(self, predictions: np.ndarray):
        n_models, n_samples, n_features = predictions.shape
        reordered = np.zeros((n_samples, n_models, n_features))
        for i in range(n_models):
            for j in range(n_samples):
                reordered[j][i] = predictions[i][j]

        return reordered

    def _compute_predictions(self, dataset: Dataset, preprocessor: ImagePreprocessor = None, sess: tf.Session = None):
        preds = self.prediction_models[0].predict(dataset, preprocessor=preprocessor, sess=sess)
        combined_preds = np.expand_dims(preds, axis=0)

        for i in range(1, len(self.prediction_models)):
            model = self.prediction_models[i]
            preds = model.predict(dataset, preprocessor=preprocessor, sess=sess)
            preds = np.expand_dims(preds, axis=0)
            combined_preds = np.concatenate([combined_preds, preds], axis=0)
        
        reordered = self._reorder_predictions(combined_preds)

        return reordered

    def _train_on_dataset(self, dataset: Dataset, *, epochs: int, preprocessor: ImagePreprocessor = None, sess: tf.Session = None):
        for model in self.prediction_models:
            model.train(dataset, preprocessor=preprocessor, epochs=epochs[0], sess=sess)

        preds = self._compute_predictions(dataset, preprocessor=preprocessor, sess=sess)

        with TFSession(sess, self.decision_model.graph) as sess:
            self.decision_model.train(preds, dataset.y, epochs=epochs[1], sess=sess)

    def _train_on_folds(self, dataset: Dataset, *, epochs: int, preprocessor: ImagePreprocessor = None, sess: tf.Session = None):
        num_models = len(self.prediction_models)
        folds = dataset.folds(num_models)

        # Create training and validation sets
        training_sets = []
        prediction_sets = []
        for i in range(len(folds)):
            training_sets.append(functools.reduce(lambda x, y: x + y, folds[:i] + folds[i + 1:],))
            prediction_sets.append(folds[i])

        # Create predictions by training the individual models on folds
        preds = []
        for i in range(num_models):
            model = self.prediction_models[i]
            for j in range(len(folds)):
                model.train(training_sets[j], epochs=epochs[0], preprocessor=preprocessor, sess=sess)
                model_preds = model.predict(prediction_sets[j], preprocessor=preprocessor, sess=sess)
                if j == 0:
                    preds.append(model_preds)
                else:
                    preds[i] = np.concatenate([preds[i], model_preds], axis=0)
                model.reset()

        preds = np.asarray(preds)
        preds = self._reorder_predictions(preds)

        # Train the base models
        for model in self.prediction_models:
            model.train(dataset, epochs=epochs[0], preprocessor=preprocessor, sess=sess)

        # Train the decision model
        dataset = Dataset(X=preds, y=dataset.y)
        with TFSession(sess, self.decision_model.graph) as sess:
            self.decision_model.train(dataset, epochs=epochs[1], sess=sess)

    def _train_on_shuffled_folds(self, dataset: Dataset, *, epochs: int, preprocessor: ImagePreprocessor = None, sess: tf.Session = None):
        log_and_raise(NotImplementedError, 'Does not work yet')
        # TODO (20.06.17): Should be combined with _train_on_folds
        num_models = len(self.prediction_models)

        datasets = []
        idxs = []
        X = dataset.X
        y = dataset.y
        for i in range(num_models):
            idx = np.random.permutation(len(dataset))
            datasets.append(dataset[idx])
            idxs.append(idx)

        preds = []
        for i in range(num_models):
            model = self.prediction_models[i]
            folds = datasets[i].folds(num_models)
            points_per_fold = math.ceil(len(dataset) / num_models)
            model_preds = []

            for j in range(len(folds)):
                training_set = functools.reduce(lambda x, y: x + y, folds[:i] + folds[i + 1:],)
                prediction_set = folds[i]
                model.train(training_set, epochs=epochs[0], preprocessor=preprocessor, sess=sess)

                model_preds = model.predict(prediction_set, preprocessor=preprocessor, sess=sess)
                if j == 0:
                    preds.append(model_preds)
                else:
                    preds[i] = np.concatenate([preds[i], model_preds], axis=0)
                model.reset()

            preds[i] = preds[i][np.argsort(idxs[i])]

        preds = np.asarray(preds)
        preds = self._reorder_predictions(preds)

        for model in self.prediction_models:
            model.train(dataset, epochs=epochs[0], preprocessor=preprocessor, sess=sess)

        dataset = Dataset(X=preds, y=dataset.y)
        with TFSession(sess, self.decision_model.graph) as sess:
            self.decision_model.train(dataset, epochs=epochs[1], sess=sess)

    def train(self, dataset: Dataset, *, epochs: int, preprocessor: ImagePreprocessor = None, sess: tf.Session = None):
        if type(epochs) is int:
            epochs = [epochs, epochs]
        elif type(epochs) is list:
            pass
        else:
            log_and_raise(InvalidArgumentException, 'Invalid epochs type %s. (Valid is [int, list])' % repr(type(epochs)))

        if self._policy is None:
            self._train_on_dataset(dataset, epochs=epochs, preprocessor=preprocessor, sess=sess)
        elif self._policy is self.DATA_POLICY_FOLDS:
            self._train_on_folds(dataset, epochs=epochs, preprocessor=preprocessor, sess=sess)
        elif self._policy is self.DATA_POLICY_SHUFFLED_FOLDS:
            self._train_on_shuffled_folds(dataset, epochs=epochs, preprocessor=preprocessor, sess=sess)
        else:
            log_and_raise(InvalidArgumentException, 'Invalid data policy %s. (Valid is %s)' % (policy, str(self._VALID_DATA_POLICIES)))
        
    def validate(self, dataset: Dataset, *, preprocessor: ImagePreprocessor = None, sess: tf.Session = None):
        preds = self._compute_predictions(dataset, preprocessor=preprocessor, sess=sess)

        for model in self.prediction_models:
            _, acc = model.validate(dataset, preprocessor=preprocessor, sess=sess)
            logger.info('%s acc: %.2f%%' % (model.name, (acc * 100)))

        dataset = Dataset(X=preds, y=dataset.y)
        variables = self.decision_model.variables
        with TFSession(sess, self.decision_model.graph, variables=variables) as sess:
            return self.decision_model.validate(dataset, sess=sess)

    def predict(self, dataset: Dataset, preprocessor: ImagePreprocessor = None, sess: tf.Session = None):
        preds = self._compute_predictions(dataset, preprocessor=preprocessor, sess=sess)

        dataset = Dataset(X=preds, y=dataset.y)
        variables = self.decision_model.variables
        with TFSession(sess, self.decision_model.graph, variables=variables) as sess:
            return self.decision_model.predict(dataset, sess=sess)

    def reset(self, **kwargs):
        for model in self.prediction_models:
            model.reset()

        self.decision_model.reset()

    def save(self, path: str, sess: tf.Session = None, **kwargs):
        metadata = kwargs
        metadata['name'] = self.name
        metadata['policy'] = self._policy

        prediction_model_paths = []
        for i in range(len(self.prediction_models)):
            model = self.prediction_models[i]
            model_path = '%s_prediction_%d' % (path, i)
            model.save(model_path)
            prediction_model_paths.append(model_path)

        metadata['prediction_model_paths'] = prediction_model_paths
        decision_model_path = '%s_decision' % path
        self.decision_model.save(decision_model_path)
        metadata['decision_model_path'] = decision_model_path

        metadata_filename = '%s.%s' % (path, METADATA_SUFFIX)
        with open(metadata_filename, 'w') as f:
            f.write(json.dumps(metadata, indent=2))

    @staticmethod
    def from_tw(path: str, sess: tf.Session = None, **kwargs):
        metadata_filename = '%s.%s' % (path, METADATA_SUFFIX)
        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)

        name = metadata['name']
        policy = metadata['policy']
        prediction_models = []
        for path in metadata['prediction_model_paths']:
            prediction_models.append(TransferLearningModel.from_tw(path))

        decision_model = NeuralNet.from_tw(metadata['decision_model_path'])

        return Stacker(prediction_models, decision_model, policy=policy, name=name)