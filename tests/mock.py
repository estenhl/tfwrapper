from tfwrapper.models import BaseModel, Predictive, RegressionModel, ClassificationModel

class MockBaseModel(BaseModel):
    @property
    def graph(self):
        return super().graph

    @property
    def variables(self):
        return super().variables

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)

    def from_tw(self, *args, **kwargs):
        super().from_tw(*args, **kwargs)


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

    def validate(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def from_tw(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass


class MockClassificationModel(ClassificationModel):
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

    def validate(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def from_tw(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass