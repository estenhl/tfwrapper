from tfwrapper.models import BaseModel, Predictive, RegressionModel, ClassificationModel

class MockBaseModel(BaseModel):
    def __init__(self):
        super().__init__('MockBaseModel')

    def graph(self):
        return self._graph

    def variables(self):
        return self._variables

    def reset(self, **kwargs):
        pass

    def save(self, path, *, sess=None, **kwargs):
        pass

    def load(self, path, **kwargs):
        pass

    def from_tw(self, path: str, sess=None, **kwargs):
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

    def validate(self, X, y, *, sess=None, **kwargs):
        pass

    def predict(self, X, *, sess=None, **kwargs):
        pass


class MockClassificationModel(ClassificationModel):
    def validate(self, X, y, *, sess=None, **kwargs):
        pass

    def predict(self, X, *, sess=None, **kwargs):
        pass