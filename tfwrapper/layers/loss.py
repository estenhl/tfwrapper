from abc import ABC, abstractmethod
import tensorflow as tf

from tfwrapper.utils.decorators import deprecated
from tfwrapper.utils.exceptions import log_and_raise
from tfwrapper.utils.exceptions import InvalidArgumentException


class Loss(ABC):
    def __init__(self, name: str = 'Loss'):
        self.name = name

    @staticmethod
    def from_name(name: str):
        name = name.lower()

        mse = ['mse', 'meansquarederror', 'mean-squared-error']
        multiclass_hinge = ['multiclasshinge', 'multiclass-hinge', 'multi-class-hinge']
        squared_multiclass_hinge = ['squaredmulticlasshinge', 'squared-multiclass-hinge', 'squared-multi-class-hinge']
        binary_hinge = ['binaryhinge', 'binary-hinge']
        squared_binary_hinge = ['squaredbinaryhinge', 'squared-binary-hinge']
        mean_softmax_crossentropy = ['meansoftmaxcrossentropy', 'mean-softmax-crossentropy', 'mean-softmax-cross-entropy']
        pixelwise_softmax_crossentropy = ['pixelwisesoftmaxcrossentropy', 'pixelwise_softmax_crossentropy', 'pixelwise_softmax_cross_entropy']

        combined = []
        combined += mse
        combined += multiclass_hinge
        combined += squared_multiclass_hinge
        combined += binary_hinge
        combined += squared_binary_hinge
        combined += mean_softmax_crossentropy
        combined += pixelwise_softmax_crossentropy

        if name in mse:
            return MSE()
        elif name in multiclass_hinge:
            return MultiClassHinge()
        elif name in squared_multiclass_hinge:
            return SquaredMultiClassHinge()
        elif name in binary_hinge:
            return BinaryHinge()
        elif name in squared_binary_hinge:
            return SquaredBinaryHinge()
        elif name in mean_softmax_crossentropy:
            return MeanSoftmaxCrossEntropy()
        elif name in pixelwise_softmax_crossentropy:
            return PixelwiseSoftmaxCrossEntropy()
        else:
            log_and_raise(NotImplementedError, '%s loss is not implemented. (Valid is %s)' % (name, combined))

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


class MSE(Loss):
    def _execute(self, y, preds, name):
        if y.get_shape()[0] != preds.get_shape()[0]:
            log_and_raise(InvalidArgumentException, 'MSE loss requires y and predictions to be of the same length')
        
        squared = tf.square(preds - y, name=name + '/squared')

        return tf.reduce_mean(squared, name=name)


class MultiClassHinge(Loss):
    # http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf
    TYPE_CRAMMER_SINGER = 'crammer_singer'

    def _execute(self, y, preds, name, method=TYPE_CRAMMER_SINGER, delta=10e-8):
        ones = tf.ones(tf.shape(y))
        reverse_onehot = tf.subtract(ones, y, name=name + '/reverse_onehot')

        w_y = tf.multiply(y, preds, name=name + '/w_y/multiply')
        w_y = tf.map_fn(lambda x: tf.reduce_sum(x), w_y, name=name + '/w_y/reduce')
        
        w_t = tf.multiply(reverse_onehot, preds, name=name + '/w_y/multiply')
        w_t = tf.map_fn(lambda x: tf.reduce_max(x), w_t, name=name + '/w_y/reduce')

        diff = tf.subtract(w_t + delta, w_y, name=name + '/individual_losses/diff')
        flat_ones = tf.ones(tf.shape(diff), name=name + '/flat_ones')
        add = tf.add(flat_ones, diff, name=name + '/add')
        flat_zeros = tf.zeros(tf.shape(add), name=name + '/flat_zeros')
        floored = tf.maximum(flat_zeros, add, name=name + '/floored')

        return tf.reduce_mean(floored, name=name)


class SquaredMultiClassHinge(Loss):
    def _execute(self, y, preds, name, **kwargs):
        return tf.square(MultiClassHinge()._execute(y, preds, name + '/provisional', **kwargs), name=name)


class BinaryHinge(Loss):
    def _execute(self, y, preds, name):
        # Transform [0, 1] y matrix to [-1, 1]
        # TODO (03.07.17): Should check that values are actually in [0, 1]
        scaled = tf.multiply(y, 2.0, name=name + '/scaled')
        shifted = tf.subtract(scaled, 1.0, name=name + '/shifted')

        # Calculate binary hinge loss
        multiplied = tf.multiply(shifted, preds, name=name + '/multiply')
        ones = tf.map_fn(lambda x: 1.0, y, name=name + '/ones')
        zeros = tf.subtract(ones, 1.0, name=name + '/zeros')
        subtracted = tf.subtract(ones, multiplied, name=name + '/subtracted')
        floored = tf.maximum(zeros, subtracted, name=name + '/floored')

        return tf.reduce_mean(floored, name=name)


class SquaredBinaryHinge(Loss):
    def _execute(self, y, preds, name):
        return tf.square(BinaryHinge()._execute(y, preds, name=name + '/provisional'), name=name)


class MeanSoftmaxCrossEntropy(Loss):
    def _execute(self, y, preds, name):
        if y is None and preds is None:
            return mean_softmax_cross_entropy

        if name is None:
            name = 'mean_softmax_cross_entropy'

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y, name=name + '/cross_entropy')
        return tf.reduce_mean(cross_entropy, name=name)


class PixelwiseSoftmaxCrossEntropy(Loss):
    def _execute(self, y, preds, name):
        num_classes = tf.shape(y)[-1]
        reshaped_y = tf.reshape(y, [-1, num_classes], name=name + '/reshaped_y')
        reshaped_preds = tf.reshape(preds, [-1, num_classes], name=name + '/reshaped_preds')
        individual_losses = tf.nn.softmax_cross_entropy_with_logits(labels=reshaped_y, logits=reshaped_preds, name=name + '/individual_losses')
        
        return tf.reduce_mean(individual_losses, name=name)


@deprecated
def mse(*, y, preds, name='mse'):
    return MSE()(y=y, preds=preds, name=name)


@deprecated
def multiclass_hinge(*, y, preds, method=MultiClassHinge.TYPE_CRAMMER_SINGER, delta=10e-8, name='hinge'):
    return MultiClassHinge()(y=y, preds=preds, name=name, method=method, delta=delta)


@deprecated
def squared_multiclass_hinge(*, y, preds, method=MultiClassHinge.TYPE_CRAMMER_SINGER, delta=10e-8, name='hinge'):
    return SquaredMultiClassHinge()(y=y, preds=preds, name=name, method=method, delta=delta)


@deprecated
def binary_hinge(*, y, preds, name='hinge'):
    return BinaryHinge()(y=y, preds=preds, name=name)


@deprecated
def squared_binary_hinge(*, y, preds, name='hinge'):
    return SquaredBinaryHinge()(y=y, preds=preds, name=name)


@deprecated
def mean_softmax_cross_entropy(*, y, preds, name):
    return MeanSoftmaxCrossEntropy()(y=y, preds=preds, name=name)


@deprecated
def pixelwise_softmax_cross_entropy(*, y, preds, name):
    return PixelwiseSoftmaxCrossEntropy()(y=y, preds=preds, name=name)





