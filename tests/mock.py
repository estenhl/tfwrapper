import tensorflow as tf

from tfwrapper.models import BaseModel, ClassificationModel, FixedRegressionModel, FixedClassificationModel, FrozenModel, RegressionModel, Predictive

class MockBaseModel(BaseModel):
    def mock(self):
        pass


class MockFixedRegressionModel(FixedRegressionModel):
    def validate(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class MockFixedClassificationModel(FixedClassificationModel):
    def validate(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class MockRegressionModel(RegressionModel):
    _graph = None
    _variables = None

    @property
    def graph(self):
        return self._graph

    @property
    def variables(self):
        return self._variables

    def __init__(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

    def load_from_meta_graph(self, *args, **kwargs):
        pass

    def from_tw(self, *args, **kwargs):
        pass

    def validate(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass


class MockClassificationModel(ClassificationModel):
        
    @classmethod
    def from_tw(cls, *args, **kwargs):
        name = kwargs['name']
        del kwargs['name']
        layers = [lambda x: tf.Variable([[5.]], name=name + '/pred')]
        return ClassificationModel.from_tw(*args, layers=layers, **kwargs)

    def validate(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass


class MockFrozenModel(FrozenModel):
    input_tensor = None
    output_tensor = None
    bottleneck_tensor = None

    def __init__(self, path, input_tensor='input', output_tensor='output', bottleneck_tensor='bottleneck', name='name', sess=None):
        FrozenModel.__init__(self, path, name=name, sess=sess)

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.bottleneck_tensor = bottleneck_tensor
