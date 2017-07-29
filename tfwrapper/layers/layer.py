class Layer():
    def __init__(self, tensor, dependencies=None):
        self.tensor = tensor
        self.dependencies = dependencies

    def __call__(self, *args, **kwargs):
        return self.tensor(*args, **kwargs)