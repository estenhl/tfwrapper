import tensorflow as tf
from abc import ABC, abstractmethod

class Accuracy(ABC):
    def __init__(self, name: str = 'Loss'):
        self.name = name

    @staticmethod
    def from_name(name: str):
        name = name.lower()

        correct_pred = ['correctpred', 'correct-pred']

        combined = []
        combined += correct_pred

        if name in correct_pred:
            return CorrectPred()
        else:
            log_and_raise(NotImplementedError, '%s accuracy is not implemented. (Valid is %s)' % (name, combined))

    def __call__(self, *, y=None, preds=None, **kwargs):
        if y is None and preds is None:
            return self

        if 'name' in kwargs:
            name = kwargs['name']
            del kwargs['name']
        else:
            name = self.name

        return self._execute(y, preds, name, **kwargs)

    @abstractmethod
    def _execute(self, y, preds, name, **kwargs):
        pass


class CorrectPred(Accuracy):
    def _execute(self, y, preds, name):
        correct_preds = tf.equal(tf.argmax(preds, 1, name=name + '/pred_argmax'), tf.argmax(y, 1, name='y_argmax'), name=name + '/correct')
        return tf.reduce_mean(tf.cast(correct_preds, tf.float32), name=name)
