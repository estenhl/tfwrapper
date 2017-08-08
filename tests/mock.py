from tfwrapper.models import BaseModel, Predictive, FixedRegressionModel, FixedClassificationModel, RegressionModel, ClassificationModel

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
    def validate(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass