import json
import datetime
import numpy as np
import tensorflow as tf
from abc import abstractmethod, abstractproperty, ABC, ABCMeta
from typing import Any, Callable, Dict, List, Union

from tfwrapper import logger, METADATA_SUFFIX, TFSession
from tfwrapper.layers import Layer
from tfwrapper.layers.accuracy import Accuracy, CorrectPred
from tfwrapper.layers.loss import Loss, MeanSoftmaxCrossEntropy, MSE
from tfwrapper.layers.optimizers import Adam, Optimizer
from tfwrapper.utils import get_variable_by_name
from tfwrapper.utils.data import get_subclass_by_name
from tfwrapper.utils.exceptions import log_and_raise, InvalidArgumentException


META_GRAPH_SUFFIX = 'meta'


_tensor_wrapper = Callable[[Union[tf.Tensor, List[tf.Tensor]]], tf.Tensor]


def _parse_layer_list(start: tf.Tensor, remaining: List[Union[tf.Tensor, _tensor_wrapper]], sess: tf.Session = None):
    tensors = [{'name': start.name, 'tensor': start}]

    prev = start
    for layer in remaining:
        if type(layer) is Layer:
            if layer.dependencies is not None:
                dependencies = layer.dependencies

                if type(dependencies) is str:
                    prev = sess.graph.get_tensor_by_name('/'.join([self.name, dependencies]) + ':0')
                elif type(dependencies) is list:
                    prev = [sess.graph.get_tensor_by_name('/'.join([self.name, dependency]) + ':0') for dependency in dependencies]
                else:
                    log_and_raise(InvalidArgumentException, 'Invalid layer dependency type %s. (Valid is [str, list])' % type(dependencies))
        elif not callable(layer):
            log_and_raise(InvalidArgumentException, 'Invalid layer type %s. (Valid is [Layer, callable]' % str(type(layer)))

        prev = layer(prev)
        tensors.append({'name': prev.name, 'tensor': prev})
        
    return tensors, prev


class BaseModel(ABC):
    """ The base class for singular machine learning models. Mainly contains infrastructure stuff """

    @property
    def graph(self) -> tf.Graph:
        """ The tf.Graph containing all tensors and variables necessary for the model """
        return self._graph

    @property
    def variables(self) -> Dict[str, tf.Tensor]:
        """ A map of all the variable values stored in the tensors of the graph. Typically trained weight and bias values.
        Used for training the model in one session, then restoring the model on a separate session """
        return self._variables
    
    def __init__(self, X_shape: List[int], y_shape: Union[int, List[int]], layers: List[Union[tf.Tensor, _tensor_wrapper]], *, preprocessing: List[Union[tf.Tensor, _tensor_wrapper]] = None, sess: tf.Session = None, name: str = 'BaseModel'):
        if preprocessing is None:
            preprocessing = []

        self.name = name
        self.X_shape = X_shape
        if type(y_shape) is int:
            self.y_shape = [y_shape]
        elif type(y_shape) is list:
            self.y_shape = y_shape
        else:
            log_and_raise(InvalidArgumentException, 'Invalid type %s for y_size. (Valid is [int, list])' % str(type(y_size)))

        with TFSession(sess) as sess:
            self.X = tf.placeholder(tf.float32, [None] + self.X_shape, name=self.name + '/X_placeholder')
            self.y = tf.placeholder(tf.float32, [None] + self.y_shape, name=self.name + '/y_placeholder')

            self.layers = preprocessing + layers
            self.tensors, self.preds = _parse_layer_list(self.X, self.layers)
            self._graph = sess.graph

        self._variables = {}
        self.init_vars_when_training = True
        self.feed_dict = {}

    @classmethod
    def from_tw(cls, filename: str, sess: tf.Session = None, **kwargs):
        """ Loads a model from the tfwrapper-specific *.tw meta-files """

        if filename.endswith('.tw'):
            metadata_filename = filename
            weights_filename = filename[:-3]
        else:
            metadata_filename = '%s.%s' % (filename, 'tw')
            weights_filename = filename

        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)

        classname = metadata['type']

        # If the subclass has not previously been imported, every known subclass is imported
        try:
            subclass = get_subclass_by_name(BaseModel, classname)
        except InvalidArgumentException:
            import nets
            import linear
            subclass = get_subclass_by_name(BaseModel, classname)

        with TFSession(sess) as sess:
            return subclass.from_tw_data(metadata, weights_filename, **kwargs, sess=sess)


    @classmethod
    def from_tw_data(cls, data: Dict[str, Any], weights_filename: str, load_params: Dict[str, Any] = None, sess: tf.Session = None, **kwargs):
        """ Loads a model from a dictionary of key-value pairs (as specified by the tfwrapper-specific *.tw files) """

        if load_params is None:
            load_params = {}

        X_shape = data['X_shape']
        y_shape = data['y_shape']
        name = data['name']

        with TFSession(sess) as sess:
            model = cls(X_shape, y_shape, **kwargs, sess=sess, name=name)
            model.load(weights_filename, **load_params, sess=sess)

            return model

    def _checkpoint_variables(self, sess):
        """ Stores values of the variables of a model (Allows for switching between sessions). """

        for variable in tf.trainable_variables():
            self._variables[variable.name] = {'tensor': variable, 'value': sess.run(variable)}

    def assign_variable_value(self, name, value, sess=None):
        """ Assigns a value to a variable. If the variable (identified by name) does not exist on the models graph,
        an InvalidArgumentException is raised """
        
        with TFSession(sess, self.graph) as sess:
            variable = get_variable_by_name(name)

            if variable is None:
                log_and_raise(InvalidArgumentException, '%s contains has no variable named %s' % (self.name, name))

            sess.run(variable.assign(value))

    def reset(self, **kwargs):
        """ Resets the value of all variables stored on the graph """

        with TFSession(graph=self._graph, variables=self._variables) as sess:
            sess.run(tf.global_variables_initializer())

    def save(self, path: str, *, sess: tf.Session = None, **kwargs):
        """ Saves the graph to a file using tf.train.Saver. Also stores a more verbose *.tw file
        which will also write additional info given in **kwargs (e.g. accuracy or labels) """

        with TFSession(sess, graph=self._graph, variables=self._variables) as sess:
            saver = tf.train.Saver(tf.trainable_variables())
            saver.save(sess, path, meta_graph_suffix=META_GRAPH_SUFFIX)

            metadata = kwargs
            metadata['name'] = self.name
            metadata['X_shape'] = self.X_shape
            metadata['y_shape'] = self.y_shape
            metadata['time'] = str(datetime.datetime.now())
            metadata['type'] = self.__class__.__name__

            metadata_filename = '%s.%s' % (path, METADATA_SUFFIX)
            with open(metadata_filename, 'w') as f:
                f.write(json.dumps(metadata, indent=2))

    def load(self, path: str, sess: tf.Session = None, **kwargs):
        """ Loads a model from a tensorflow *.data* file. Requires the structure of the model
        to match the structure in the file """

        with TFSession(sess, graph=self._graph) as sess:
            sess.run(tf.variables_initializer(tf.trainable_variables()))
            saver = tf.train.Saver(tf.trainable_variables())
            saver.restore(sess, path)

            self.X = sess.graph.get_tensor_by_name(self.name + '/X_placeholder:0')
            self.y = sess.graph.get_tensor_by_name(self.name + '/y_placeholder:0')
            self.preds = sess.graph.get_tensor_by_name(self.name + '/pred:0')

            self._checkpoint_variables(sess)
            self._graph = sess.graph

    def load_from_meta_graph(self, path: str, sess: tf.Session = None):
        """ Loads a model from a tensorflow *.meta file. Requires the structure of the model
        to match the structure in the file """

        with TFSession(sess) as sess:
            if not path.endswith(META_GRAPH_SUFFIX):
                path = '%s.%s' % (path, META_GRAPH_SUFFIX)

            print(path)

            new_saver = tf.train.import_meta_graph(path, clear_devices=True)
            new_saver.restore(sess, path)

            self.X = graph.get_tensor_by_name(self.name + '/X_placeholder:0')
            self.y = graph.get_tensor_by_name(self.name + '/y_placeholder:0')
            self.preds = graph.get_tensor_by_name(self.name + pred_tensor_name)

            self.X_shape = self.X.shape
            self.input_size = np.prod(self.X_shape)
            self.y_shape = self.y.shape

            self._checkpoint_variables(sess)
            self._graph = sess.graph


class Predictive(ABC):
    @abstractmethod
    def validate(self, X: np.ndarray, y: np.ndarray, *, sess: tf.Session = None, **kwargs):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, *, sess: tf.Session = None, **kwargs) -> np.ndarray:
        pass


class PredictiveModel(BaseModel, Predictive):
    DEFAULT_LOSS = MSE

    @property
    def loss(self) -> tf.Tensor:
        return self._loss

    @loss.setter
    def loss(self, value: Union[tf.Tensor, Loss, ABCMeta, Callable[..., tf.Tensor]]):
        with TFSession(None, graph=self.graph, variables=self.variables):
            if type(value) is tf.Tensor:
                self._loss = value
            elif isinstance(value, Loss):
                self._loss = value(y=self.y, preds=self.preds, name=self.name + '/loss')
            elif isinstance(value, ABCMeta):
                self._loss = value()(y=self.y, preds=self.preds, name=self.name + '/loss')
            elif callable(value):
                self._loss = value(y=self.y, preds=self.preds, name=self.name + '/loss')
            elif type(value) is str:
                f = Loss.from_name(value)
                self._loss = f(y=self.y, preds=self.preds, name=self.name + '/loss')
            else:
                log_and_raise(InvalidArgumentException, 'Invalid loss type %s. (Valid is [tf.Tensor, Loss, ABCMeta, callable, str])' % str(type(value)))

            self._loss_tensor_name = self._loss.name

    def __init__(self, X_shape: List[int], y_size: Union[int, List[int]], layers: List[Union[tf.Tensor, _tensor_wrapper]], *, preprocessing: List[Union[tf.Tensor, _tensor_wrapper]] = None, sess: tf.Session = None, name: str = 'BaseModel'):
        with TFSession(sess) as sess:
            BaseModel.__init__(self, X_shape, y_size, layers, preprocessing=preprocessing, sess=sess, name=name)
            self.loss = self.DEFAULT_LOSS

    @classmethod
    def from_tw_data(cls, data: Dict[str, Any], weights_filename: str, load_params: Dict[str, Any] = None, sess: tf.Session = None, **kwargs):
        if load_params is None:
            load_params = {}

        if 'loss_tensor_name' in data:
            load_params['loss_tensor_name'] = data['loss_tensor_name']
            del data['loss_tensor_name']

        return super().from_tw_data(data, weights_filename, load_params, sess, **kwargs)

    def save(self, path: str, sess: tf.Session = None, **kwargs):
        with TFSession(sess, graph=self.graph, variables=self.variables) as sess:
            BaseModel.save(self, path, loss_tensor_name=self._loss_tensor_name, sess=sess, **kwargs)

    def load(self, path: str, loss_tensor_name: str = None, sess: tf.Session = None, **kwargs):
        if loss_tensor_name is None:
            loss_tensor_name = self.name + '/loss:0'

        with TFSession(sess) as sess:
            # If the loss-tensor does not exist, create a virtual no_op that can load the tensor from file
            try:
                sess.graph.get_tensor_by_name(loss_tensor_name)
            except Exception:
                # Drops the ':X' part of the name
                tf.multiply(1, 1, name=loss_tensor_name[:-2])

            BaseModel.load(self, path, sess=sess, **kwargs)
            self._loss = self.graph.get_tensor_by_name(loss_tensor_name)

    def load_from_meta_graph(self, path: str, sess: tf.Session = None):
        with TFSession(sess) as sess:
            BaseModel.load_from_meta_graph(self, path, sess=sess)
            self._loss = self.graph.get_tensor_by_name(self.name + '/loss:0')


class FixedRegressionModel(PredictiveModel):
    @abstractmethod
    def validate(self, X: np.ndarray, y: np.ndarray, *, sess: tf.Session = None, **kwargs) -> float:
        pass


class FixedClassificationModel(PredictiveModel):
    DEFAULT_LOSS = MeanSoftmaxCrossEntropy
    DEFAULT_ACCURACY = CorrectPred

    @property
    def accuracy(self) -> tf.Tensor:
        return self._accuracy

    @accuracy.setter
    def accuracy(self, value: Union[tf.Tensor, Accuracy, ABCMeta, Callable[..., tf.Tensor]]):
        with TFSession(None, graph=self.graph, variables=self.variables):
            if type(value) is tf.Tensor:
                self._accuracy = value
            elif isinstance(value, Accuracy):
                self._accuracy = value(y=self.y, preds=self.preds, name=self.name + '/accuracy')
            elif isinstance(value, ABCMeta):
                self._accuracy = value()(y=self.y, preds=self.preds, name=self.name + '/accuracy')
            elif callable(value):
                self._accuracy = value(y=self.y, preds=self.preds, name=self.name + '/accuracy')
            elif type(value) is str:
                f = Accuracy.from_name(value)
                self._accuracy = f(y=self.y, preds=self.preds, name=self.name + '/accuracy')
            else:
                log_and_raise(InvalidArgumentException, 'Invalid accuracy type %s. (Valid is [tf.Tensor, Accuracy, ABCMeta, callable, str])' % str(type(value)))

            self._accuracy_tensor_name = self._accuracy.name
    
    def __init__(self, X_shape: List[int], y_size: Union[int, List[int]], layers: List[Union[tf.Tensor, _tensor_wrapper]], *, preprocessing: List[Union[tf.Tensor, _tensor_wrapper]] = None, sess: tf.Session = None, name: str = 'BaseModel'):
        with TFSession(sess) as sess:
            PredictiveModel.__init__(self, X_shape, y_size, layers, preprocessing=preprocessing, sess=sess, name=name)
            self.accuracy = self.DEFAULT_ACCURACY

    @classmethod
    def from_tw_data(cls, data: Dict[str, Any], weights_filename: str, load_params: Dict[str, Any] = None, sess: tf.Session = None, **kwargs):
        if load_params is None:
            load_params = {}

        if 'accuracy_tensor_name' in data:
            load_params['accuracy_tensor_name'] = data['accuracy_tensor_name']
            del data['accuracy_tensor_name']

        return super().from_tw_data(data, weights_filename, load_params, sess, **kwargs)

    def save(self, path: str, sess: tf.Session = None, **kwargs):
        with TFSession(sess, graph=self.graph, variables=self.variables) as sess:
            PredictiveModel.save(self, path, accuracy_tensor_name=self._accuracy_tensor_name, sess=sess, **kwargs)

    def load(self, path: str, accuracy_tensor_name: str = None, sess: tf.Session = None, **kwargs):
        if accuracy_tensor_name is None:
            accuracy_tensor_name = self.name + '/accuracy:0'

        with TFSession(sess) as sess:
            # If the accuracy-tensor does not exist, create a virtual no_op that can load the tensor from file
            try:
                sess.graph.get_tensor_by_name(accuracy_tensor_name)
            except Exception:
                # Drops the ':X' part of the name
                tf.multiply(1, 1, name=accuracy_tensor_name[:-2])

            PredictiveModel.load(self, path, sess=sess, **kwargs)
            self._accuracy = self.graph.get_tensor_by_name(accuracy_tensor_name)

    def load_from_meta_graph(self, path: str, sess: tf.Session = None):
        with TFSession(sess) as sess:
            PredictiveModel.load_from_meta_graph(self, path, sess=sess)
            self._accuracy = self.graph.get_tensor_by_name(self.name + '/accuracy:0')

    @abstractmethod
    def validate(self, X: np.ndarray, y: np.ndarray, *, sess: tf.Session = None, **kwargs) -> (float, float):
        pass


class Trainable(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, *, epochs: int, sess: tf.Session = None, **kwargs):
        pass


class RegressionModel(FixedRegressionModel, Trainable):
    def hei():
        print('hå')


class ClassificationModel(FixedClassificationModel, Trainable):
    DEFAULT_OPTIMIZER = Adam
    DEFAULT_LEARNING_RATE = 0.01
    DEFAULT_BATCH_SIZE = 128

    @property
    def learning_rate(self) -> Union[float, Callable[..., float]]:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: Union[float, Callable[..., float]]):
        self._learning_rate = value

    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value

    @property
    def optimizer(self) -> tf.Tensor:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: Union[tf.Operation, Optimizer, ABCMeta, Callable[..., tf.Tensor]]):
        with TFSession(None, graph=self.graph, variables=self.variables):
            if hasattr(self, '_optimizer') and self._optimizer is not None:
                logger.warning('Changing optimizer currently breaks save and load due to naming')

            self._optimizer_key = value
            if type(value) is tf.Operation:
                self._optimizer = value
            elif isinstance(value, Optimizer):
                self._optimizer = value(learning_rate=self.lr_placeholder, loss=self.loss, name=self.name + '/optimizer')
            elif isinstance(value, ABCMeta):
                self._optimizer = value()(learning_rate=self.lr_placeholder, loss=self.loss, name=self.name + '/optimizer')
            elif callable(value):
                self._optimizer = value(learning_rate=self.lr_placeholder, loss=self.loss, name=self.name + '/optimizer')
            elif type(value) is str:
                f = Optimizer.from_name(value)
                self._optimizer = f(learning_rate=self.lr_placeholder, loss=self.loss, name=self.name + '/optimizer')
            else:
                log_and_raise(InvalidArgumentException, 'Invalid optimizer type %s. (Valid is [tf.Operation, Optimizer, ABCMeta, callable, str])' % str(type(value)))

            self._optimizer_tensor_name = self._optimizer.name

    @property
    def loss(self) -> tf.Tensor:
        return FixedClassificationModel.loss.fget(self)
    
    @loss.setter
    def loss(self, value: Union[tf.Tensor, Loss, ABCMeta, Callable[..., tf.Tensor]]):
        FixedClassificationModel.loss.fset(self, value)
        self.optimizer = self._optimizer_key

    def __init__(self, X_shape: List[int], y_size: Union[int, List[int]], layers: List[Union[tf.Tensor, _tensor_wrapper]], *, preprocessing: List[Union[tf.Tensor, _tensor_wrapper]] = None, sess: tf.Session = None, name: str = 'ClassificationModel'):
        with TFSession(sess) as sess:
            self._optimizer_key = self.DEFAULT_OPTIMIZER
            self.lr_placeholder = tf.placeholder_with_default(self.DEFAULT_LEARNING_RATE, [], name=name + '/lr_placeholder')
            self.learning_rate = self.DEFAULT_LEARNING_RATE
            self.batch_size = self.DEFAULT_BATCH_SIZE

            FixedClassificationModel.__init__(self, X_shape, y_size, layers, preprocessing=preprocessing, sess=sess, name=name)

    def save(self, *args, **kwargs):
        super().save(*args, optimizer_tensor_name=self._optimizer_tensor_name, learning_rate=self._learning_rate, batch_size=self._batch_size, **kwargs)


class Derivable(ABC):
    @abstractmethod
    def extract_features(self, layer: str = None, *, sess: tf.Session = None, **kwargs) -> np.ndarray:
        pass

