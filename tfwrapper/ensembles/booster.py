import numpy as np

from tfwrapper import logger
from tfwrapper.metrics import loss
from tfwrapper.metrics import accuracy

from .utils import accumulate_predictions
from .utils import vote_predictions

class Booster():
    def __init__(self, models):
        self.models = models

    def train(self, dataset, *, epochs):
        folds = dataset.folds(len(self.models))

        for i in range(len(folds)):
            self.models[i].train(folds[i].X, folds[i].y, epochs=epochs)

    def predict(self, X, method='accumulate'):
        preds = []

        for i in range(len(self.models)):
            preds.append(self.models[i].predict(X))


        preds = np.asarray(preds)

        if method == 'accumulate':
            return accumulate_predictions(preds)
        elif method == 'majority':
            return vote_predictions(preds)
        else:
            errormsg = 'Invalid method for combining predictions %s. (Valid is [\'accumulate\'], [\'majority\'])' % method
            logger.error(errormsg)
            raise InvalidArgumentException(errormsg)

    def validate(self, X, y, method='accumulate'):
        preds = self.predict(X, method=method)
        l = loss(y, preds)
        a = accuracy(y, preds)

        return l, a