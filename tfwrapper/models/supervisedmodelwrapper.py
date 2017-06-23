from .metamodel import MetaModel

class SupervisedModelWrapper(MetaModel):
    def __init__(self, model: SupervisedModel, name='SupervisedModelWrapper'):
        super().__init__(name)
        self.model = model

    def train(self, dataset, *, epochs, preprocessor=None, sess=None, **kwargs):
        if preprocessor is not None:
            dataset.preprocessor = preprocessor

        X, y = dataset.X, dataset.y
        self.model.train(X=X, y=y, epochs=epochs, sess=sess, **kwargs)

    @abstractmethod
    def validate(self, dataset, *, preprocessor=None, sess=None, **kwargs):
        pass

    @abstractmethod
    def predict(self, dataset, *, preprocessor=None, sess=None, **kwargs):
        pass

    def reset(self, **kwargs):
        self.model.reset(**kwargs)

    @abstractmethod
    def save(self, path, sess=None, **kwargs):
        pass

    @abstractmethod
    def from_tw(path, sess=None, **kwargs):
        pass