import copy

class TransferLearningEnsemble():
    def __init__(self, pretrained, models, feature_files=None):
        self.pretrained = pretrained
        self.models = models
        self.feature_files = feature_files

    def train(self, dataset, *, epochs):
        features = []
        for model in self.models:
