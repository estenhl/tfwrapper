import pytest
import tensorflow as tf

from tfwrapper.models import ClassificationModel

from mock import MockFixedClassificationModel, MockClassificationModel
from utils import softmax_wrapper

def test_fixed_init():
    name = 'test-fixed-init'
    model = MockFixedClassificationModel([2, 2], 2, [softmax_wrapper()], name=name)

    assert model.loss is not None, 'FixedClassificationModel.__init__ does not set loss'
    assert type(model.loss) is tf.Tensor, 'FixedClassificationModel.__init__ does not set loss to a tensor'
    assert name + '/loss:0' == model.loss.name, 'FixedClassificationModel.__init__ does not set the name of the loss-tensor to loss'
    assert model.accuracy is not None, 'FixedClassificationModel.__init__ does not set accuracy'
    assert type(model.accuracy) is tf.Tensor, 'FixedClassificationModel.__init__ does not set accuracy to a tensor'
    assert name + '/accuracy:0' == model.accuracy.name, 'FixedClassificationModel.__init__ does not set the name of the accuracy-tensor to accuracy'


def test_set_loss():
    name = 'test-set-loss'
    model = MockFixedClassificationModel([2, 2], 1, [lambda x: tf.Variable([[.2]])], name=name)
    model.loss = lambda y, preds, name: tf.divide(y, preds, name=name)

    with tf.Session(graph=model.graph) as sess:
        result = sess.run(model.loss, feed_dict={model.y: [[2.]], model.preds: [[4.]]})

    assert len(result.shape) == 2
    assert result[0][0] == 0.5
    # TODO (08.08.17): Reinstate when we figure out how
    #assert name + '/loss:0' == model.loss.name


def test_set_accuracy():
    name = 'test-set-accuracy'
    model = MockFixedClassificationModel([2, 2], 1, [lambda x: tf.Variable([[.2]])], name=name)
    model.accuracy = lambda y, preds, name: tf.multiply(y, preds, name=name)

    with tf.Session(graph=model.graph) as sess:
        result = sess.run(model.accuracy, feed_dict={model.y: [[2.]], model.preds: [[4.]]})

    assert len(result.shape) == 2
    assert result[0][0] == 8
    # TODO (08.08.17): Reinstate when we figure out how
    #assert name + '/accuracy:0' == model.accuracy.name


def test_init():
    name = 'test-init'
    model = MockClassificationModel([2, 2], 2, [lambda x: tf.Variable([5., 5.])], name=name)

    assert model.optimizer is not None, 'FixedClassificationModel.__init__ does not set optimizer'
    assert type(model.optimizer) is tf.Operation, 'FixedClassificationModel.__init__ does not set optimizer to a tensor'
    assert name + '/optimizer' == model.optimizer.name, 'FixedClassificationModel.__init__ does not set the name of the optimizer-tensor to optimizer'
