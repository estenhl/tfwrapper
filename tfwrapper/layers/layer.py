class Layer():
    def __init__(self, tensor, dependencies=None, **init_args):
        self.tensor = tensor
        self.dependencies = dependencies
        self.init_args = init_args

    def __call__(self, *args, **kwargs):
        return self.tensor(*args, **self.init_args, **kwargs)