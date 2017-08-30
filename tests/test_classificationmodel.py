import os
import json
import pytest
import tensorflow as tf

from tfwrapper.models import ClassificationModel
from tfwrapper.layers.accuracy import CorrectPred
from tfwrapper.layers.loss import MSE
from tfwrapper.layers.optimizers import SGD

from mock import MockFixedClassificationModel, MockClassificationModel
from utils import curr_path, remove_dir, softmax_wrapper


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
    model = MockFixedClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]])], name=name)
    model.loss = lambda y, preds, name: tf.divide(y, preds, name=name)

    with tf.Session(graph=model.graph) as sess:
        result = sess.run(model.loss, feed_dict={model.y: [[2.]], model.preds: [[4.]]})

    assert len(result.shape) == 2, 'Setting a custom loss in ClassificationModel does not return the correct shape'
    assert result[0][0] == 0.5, 'The loss sat in a ClassificationModel is not used by the model'


def test_set_accuracy():
    name = 'test-set-accuracy'
    model = MockFixedClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]])], name=name)
    model.accuracy = lambda y, preds, name: tf.multiply(y, preds, name=name)

    with tf.Session(graph=model.graph) as sess:
        result = sess.run(model.accuracy, feed_dict={model.y: [[2.]], model.preds: [[4.]]})

    assert len(result.shape) == 2, 'Setting a custom accuracy in ClassificationModel does not return the correct shape'
    assert result[0][0] == 8, 'The accuracy sat in a ClassificationModel is not used by the model'


def test_init():
    name = 'test-init'
    model = MockClassificationModel([2, 2], 2, [lambda x: tf.Variable([5., 5.])], name=name)

    assert model.optimizer is not None, 'FixedClassificationModel.__init__ does not set optimizer'
    assert type(model.optimizer) is tf.Operation, 'FixedClassificationModel.__init__ does not set optimizer to a tensor'
    assert name + '/optimizer' == model.optimizer.name, 'FixedClassificationModel.__init__ does not set the name of the optimizer-tensor to optimizer'


def test_set_optimizer():
    name = 'test-set-optimizer'
    model = MockClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]])], name=name)
    old_optimizer = model.optimizer
    with tf.Session(graph=model.graph) as sess:
        model.optimizer = 'sgd'

    assert old_optimizer != model.optimizer


def test_save_loss():
    name = 'test-save-loss'
    folder = os.path.join(curr_path, 'test')
    path = os.path.join(folder, 'model')
    try:
        os.mkdir(folder)

        with tf.Session() as sess:
            model = MockClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]])], sess=sess, name=name)
            sess.run(tf.global_variables_initializer())
            model.loss = MSE()
            model.save(path, sess=sess)

        tf.reset_default_graph()

        with open(path + '.tw', 'r') as f:
            data = json.load(f)

        assert 'loss_tensor_name' in data, 'MockClassificationModel does not store loss tensor name'
        assert model.loss.name == data['loss_tensor_name'], 'MockClassificationModel does not store correct loss tensor name'
    finally:
        remove_dir(folder)


def test_load_default_loss():
    name = 'test-load-default-loss'
    folder = os.path.join(curr_path, 'test')
    path = os.path.join(folder, 'model')
    try:
        os.mkdir(folder)

        with tf.Session() as sess:
            model = MockClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]], name=name + '/pred')], sess=sess, name=name)
            sess.run(tf.global_variables_initializer())
            model.save(path, sess=sess)

        tf.reset_default_graph()

        with tf.Session() as sess:
            loaded_model = MockClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]], name=name + '/pred')], sess=sess, name=name)
            loaded_model.load(path, sess=sess)

        assert model.loss.name == loaded_model.loss.name

    finally:
        remove_dir(folder)


def test_load_custom_loss():
    name = 'test-load-custom-loss'
    folder = os.path.join(curr_path, 'test')
    path = os.path.join(folder, 'model')
    try:
        os.mkdir(folder)

        with tf.Session() as sess:
            model = MockClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]], name=name + '/pred')], sess=sess, name=name)
            sess.run(tf.global_variables_initializer())
            model.loss = MSE()
            model.save(path, sess=sess)

        tf.reset_default_graph()

        with tf.Session() as sess:
            loaded_model = MockClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]], name=name + '/pred')], sess=sess, name=name)
            loaded_model.load(path, loss_tensor_name=model.loss.name, sess=sess)

        assert model.loss.name == loaded_model.loss.name

    finally:
        remove_dir(folder)


def test_loss_from_tw():
    name = 'test-loss-from-tw'
    folder = os.path.join(curr_path, 'test')
    path = os.path.join(folder, 'model')
    try:
        os.mkdir(folder)

        with tf.Session() as sess:
            model = MockClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]], name=name + '/pred')], sess=sess, name=name)
            sess.run(tf.global_variables_initializer())
            model.loss = MSE()
            model.save(path, sess=sess)

        tf.reset_default_graph()

        with tf.Session() as sess:
            loaded_model = MockClassificationModel.from_tw(path + '.tw', name=name)

        assert model.loss.name == loaded_model.loss.name

    finally:
        remove_dir(folder)


def test_save_accuracy():
    name = 'test-save-accuracy'
    folder = os.path.join(curr_path, 'test')
    path = os.path.join(folder, 'model')
    try:
        os.mkdir(folder)

        with tf.Session() as sess:
            model = MockClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]])], sess=sess, name=name)
            sess.run(tf.global_variables_initializer())
            model.accuracy = CorrectPred()
            model.save(path, sess=sess)

        with open(path + '.tw', 'r') as f:
            data = json.load(f)

        assert 'accuracy_tensor_name' in data, 'MockClassificationModel does not store accuracy tensor name'
        assert model.accuracy.name == data['accuracy_tensor_name'], 'MockClassificationModel does not store correct accuracy tensor name'
    finally:
        remove_dir(folder)


def test_load_default_accuracy():
    name = 'test-load-default-accuracy'
    folder = os.path.join(curr_path, 'test')
    path = os.path.join(folder, 'model')
    try:
        os.mkdir(folder)

        with tf.Session() as sess:
            model = MockClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]], name=name + '/pred')], sess=sess, name=name)
            sess.run(tf.global_variables_initializer())
            model.save(path, sess=sess)

        tf.reset_default_graph()

        with tf.Session() as sess:
            loaded_model = MockClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]], name=name + '/pred')], sess=sess, name=name)
            loaded_model.load(path, sess=sess)

        assert model.accuracy.name == loaded_model.accuracy.name

    finally:
        remove_dir(folder)


def test_load_custom_accuracy():
    name = 'test-load-custom-accuracy'
    folder = os.path.join(curr_path, 'test')
    path = os.path.join(folder, 'model')
    try:
        os.mkdir(folder)

        with tf.Session() as sess:
            model = MockClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]], name=name + '/pred')], sess=sess, name=name)
            sess.run(tf.global_variables_initializer())
            model.accuracy = CorrectPred
            model.save(path, sess=sess)

        tf.reset_default_graph()

        with tf.Session() as sess:
            loaded_model = MockClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]], name=name + '/pred')], sess=sess, name=name)
            loaded_model.load(path, accuracy_tensor_name=model.accuracy.name, sess=sess)

        assert model.accuracy.name == loaded_model.accuracy.name

    finally:
        remove_dir(folder)


def test_accuracy_from_tw():
    name = 'test-accuracy-from-tw'
    folder = os.path.join(curr_path, 'test')
    path = os.path.join(folder, 'model')
    try:
        os.mkdir(folder)

        with tf.Session() as sess:
            model = MockClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]], name=name + '/pred')], sess=sess, name=name)
            sess.run(tf.global_variables_initializer())
            model.accuracy = CorrectPred
            model.save(path, sess=sess)

        tf.reset_default_graph()

        with tf.Session() as sess:
            loaded_model = MockClassificationModel.from_tw(path + '.tw', name=name)

        assert model.accuracy.name == loaded_model.accuracy.name

    finally:
        remove_dir(folder)


def test_save_optimizer():
    name = 'test-save-optimizer'
    folder = os.path.join(curr_path, 'test')
    path = os.path.join(folder, 'model')
    try:
        os.mkdir(folder)

        with tf.Session() as sess:
            model = MockClassificationModel([2, 2], 1, [lambda x: tf.Variable([[2.]])], sess=sess, name=name)
            sess.run(tf.global_variables_initializer())
            model.optimizer = SGD
            model.save(path, sess=sess)

        with open(path + '.tw', 'r') as f:
            data = json.load(f)

        assert 'optimizer_tensor_name' in data, 'MockClassificationModel does not store optimizer tensor name'
        assert model.optimizer.name == data['optimizer_tensor_name'], 'MockClassificationModel does not store correct optimizer tensor name'
    finally:
        remove_dir(folder)

