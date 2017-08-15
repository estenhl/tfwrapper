import tensorflow as tf
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, name: str = 'Loss'):
        self.name = name

    @staticmethod
    def from_name(name: str):
        name = name.lower()

        adam = ['adam']
        sgd = ['sgd', 'stochastic-gradient-descent']

        combined = []
        combined += adam
        combined += sgd

        if name in adam:
            return Adam()
        elif name in sgd:
            return SGD()
        else:
            log_and_raise(NotImplementedError, '%s accuracy is not implemented. (Valid is %s)' % (name, combined))

    def __call__(self, *, learning_rate, loss=None, **kwargs):
        if loss is None:
            return self

        if 'name' in kwargs:
            name = kwargs['name']
            del kwargs['name']
        else:
            name = self.name

        return self._execute(learning_rate, loss, name, **kwargs)

    @abstractmethod
    def _execute(self, lr, loss, name, **kwargs):
        pass


class Adam(Optimizer):
    def _execute(self, lr, loss, name, beta1=0.9, beta2=0.999, epsilon=1e-5):
        #beta1_var = tf.Variable(beta1, name=name + '/beta1')
        #beta2_var = tf.Variable(beta2, name=name + '/beta2')
        #epsilon_var = tf.Variable(epsilon, name=name + '/epsilon')

        return tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, name=name + '/adam').minimize(loss, name=name)


class SGD(Optimizer):
    def _execute(self, lr, loss, name):
        return tf.train.GradientDescentOptimizer(learning_rate=lr, name=name + '/sgd').minimize(loss, name=name)