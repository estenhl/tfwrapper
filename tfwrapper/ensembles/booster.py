import numpy as np

from tfwrapper.metrics import loss
from tfwrapper.metrics import accuracy

class Booster():
    def __init__(self, models):
        self.models = models

    def train(self, dataset, *, epochs):
        folds = dataset.folds(len(self.models))

        for i in range(len(folds)):
            self.models[i].train(folds[i].X, folds[i].y, epochs=epochs)

    def predict(self, X):
        preds = []

        for i in range(len(self.models)):
            preds.append(self.models[i].predict(X))


        preds = np.asarray(preds)
        print('BEFORE: ' + str(preds.shape))

        return np.sum(preds, axis=0)

    def validate(self, X, y):
        preds = self.predict(X)
        l = loss(y, preds)
        a = accuracy(y, preds)

        return l, a