import json
import os
import pytest
import numpy as np
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.layers import Layer
from tfwrapper.models.basemodel import _parse_layer_list
from tfwrapper.utils.exceptions import InvalidArgumentException

from mock import MockBaseModel
from utils import curr_path, softmax_wrapper, remove_dir


def test_parse_layer_list_wrappers():
    start = tf.placeholder(tf.float32, [10])
    remaining = [softmax_wrapper(), softmax_wrapper(name='last1')]

    tensors, last = _parse_layer_list(start, remaining)

    assert 3 == len(tensors), 'All tensors are not stored in the tensors-list'
    assert 'last1:0' == last.name, 'The last tensor in the list is not returned'


def test_parse_layer_list_layers():
    start = tf.placeholder(tf.float32, [10])
    remaining = [Layer(tf.nn.softmax), Layer(tf.nn.softmax, name='last2')]

    tensors, last = _parse_layer_list(start, remaining)

    assert 3 == len(tensors), 'All tensors are not stored in the tensors-list'
    assert 'last2:0' == last.name, 'The last tensor in the list is not returned'


def test_parse_layer_list_mixed():
    start = tf.placeholder(tf.float32, [10])
    remaining = [softmax_wrapper(), Layer(tf.nn.softmax, name='last3')]

    tensors, last = _parse_layer_list(start, remaining)

    assert 3 == len(tensors), 'All tensors are not stored in the tensors-list'
    assert 'last3:0' == last.name, 'The last tensor in the list is not returned'


def test_parse_layer_list_invalid():
    start = tf.placeholder(tf.float32, [10])
    remaining = [softmax_wrapper(), 'b']

    exception = False
    try:
        tensors, last = _parse_layer_list(start, remaining)
    except InvalidArgumentException:
        exception = True

    assert exception, 'Giving an invalid layer type does not raise an exception'


def test_preprocessing_is_added():
    X_shape = [1, 2]
    y_size = 2
    layers = [softmax_wrapper()]
    preprocessing = [softmax_wrapper()]

    model = MockBaseModel(X_shape, y_size, layers, preprocessing=preprocessing)

    assert 2 == len(model.layers), 'Preprocessing layers are not added to layers list'
    assert 3 == len(model.tensors), 'Preprocessing tensors are not added to tensors lists'


def test_name():
    X_shape = [1, 2]
    y_size = 2
    layers = [softmax_wrapper()]
    preprocessing = [softmax_wrapper()]
    name = 'name'

    model = MockBaseModel(X_shape, y_size, layers, preprocessing=preprocessing, name=name)

    assert name == model.name, 'BaseModel does not store the name given in __init__'


def test_reset():
    with tf.Session() as sess:
        var = tf.random_normal([5])
        sess.run(tf.global_variables_initializer())
        old_value = sess.run(var)

        model = MockBaseModel([1, 2], [2], [softmax_wrapper()], sess=sess)
        model.reset()

        new_value = sess.run(var)

    assert not np.array_equal(old_value, new_value), 'Calling BaseModel.reset() does not reset variable values'


def test_save():
    name = 'test-save'
    folder = os.path.join(curr_path, 'test')
    path = os.path.join(folder, 'model')
    pred_value = 2
    try:
        os.mkdir(folder)
       
        with tf.Session() as sess:
            model = MockBaseModel([2, 2], 1, [lambda x: tf.Variable(pred_value, name=name + '/pred')], sess=sess, name=name)
            sess.run(tf.global_variables_initializer())
            model.save(path, sess=sess)

        with tf.Session() as sess:
            tf.train.Saver().restore(sess, path)
            pred = sess.graph.get_tensor_by_name('%s/%s' % (name, 'pred:0'))

            assert sess.graph.get_tensor_by_name('%s/%s' % (name, 'X_placeholder:0')) is not None, 'Saving a model does not save the X placeholder'
            assert sess.graph.get_tensor_by_name('%s/%s' % (name, 'y_placeholder:0')) is not None, 'Saving a model does not save the y placeholder'

            assert pred is not None, 'Saving a model does not save the layers'
            assert pred_value == sess.run(pred), 'Saving a model does not save the variables with correct values'
    finally:
        remove_dir(folder)


def test_save_without_session():
    name = 'test-save'
    folder = os.path.join(curr_path, 'test')
    path = os.path.join(folder, 'model')
    pred_value = 2
    try:
        os.mkdir(folder)
       
        
        model = MockBaseModel([2, 2], 1, [lambda x: tf.Variable(pred_value, name=name + '/pred')], name=name)
        with tf.Session(graph=model.graph) as sess:
            sess.run(tf.global_variables_initializer())
            model._checkpoint_variables(sess)
        model.save(path)

        with tf.Session() as sess:
            tf.train.Saver().restore(sess, path)
            pred = sess.graph.get_tensor_by_name('%s/%s' % (name, 'pred:0'))

            assert sess.graph.get_tensor_by_name('%s/%s' % (name, 'X_placeholder:0')) is not None, 'Saving a model does not save the X placeholder'
            assert sess.graph.get_tensor_by_name('%s/%s' % (name, 'y_placeholder:0')) is not None, 'Saving a model does not save the y placeholder'

            assert pred is not None, 'Saving a model does not save the layers'
            assert pred_value == sess.run(pred), 'Saving a model does not save the variables with correct values'
    finally:
        remove_dir(folder)


def test_load():
    name = 'test-load'
    folder = os.path.join(curr_path, 'test')
    path = os.path.join(folder, 'load-model')

    try:
        os.mkdir(folder)
       
        with tf.Session() as sess:
            model = MockBaseModel([2], 1, [lambda x: tf.multiply(x, tf.Variable(2., trainable=True, name='W'), name=name + '/pred')], sess=sess, name=name)
            sess.run(tf.global_variables_initializer())
            model.save(path, sess=sess)

        tf.reset_default_graph()

        with tf.Session() as sess:
            model = MockBaseModel([2], 1, [lambda x: tf.multiply(x, tf.Variable(3., trainable=True, name='W'), name=name + '/pred')], sess=sess, name=name)
            model.load(path, sess=sess)
            value = sess.run(model.preds, feed_dict={model.X: [[3., 3.]]})

        assert np.array_equal(np.asarray([[6, 6]]), value), 'Loading a model does not restore trainable variable values'
        assert len(model.variables) == 1, 'Loading a model does not save the loaded trainable variables'
        assert 'W:0' in model.variables, 'Loading a model does not save the loaded trainable variables with the correct name'
        assert 2. ==  model.variables['W:0']['value'], 'Loading a model does not save the loaded trainable variables with the correct value'
    finally:
        remove_dir(folder)


def test_load_without_session():
    name = 'test-load-no-sess'
    folder = os.path.join(curr_path, 'test')
    path = os.path.join(folder, 'load-model-no-sess')

    try:
        os.mkdir(folder)
        
        with tf.Session() as sess:
            model = MockBaseModel([2], 1, [lambda x: tf.multiply(x, tf.Variable(2., trainable=True, name='W2'), name=name + '/pred')], sess=sess, name=name)
            sess.run(tf.global_variables_initializer())
            model.save(path, sess=sess)

        tf.reset_default_graph()

        model = MockBaseModel([2], 1, [lambda x: tf.multiply(x, tf.Variable(3., trainable=True, name='W2'), name=name + '/pred')], name=name)
        model.load(path)
        with TFSession(graph=model.graph, variables=model.variables) as sess:
            value = sess.run(model.preds, feed_dict={model.X: [[3., 3.]]})

        assert np.array_equal(np.asarray([[6, 6]]), value), 'Loading a model without a session does not restore trainable variable values'
        assert len(model.variables) == 1, 'Loading a model without a session does not save the loaded trainable variables'
        assert 'W2:0' in model.variables, 'Loading a model without a session does not save the loaded trainable variables with the correct name'
        assert 2. ==  model.variables['W2:0']['value'], 'Loading a model without a session does not save the loaded trainable variables with the correct value'
    finally:
        remove_dir(folder)


def test_from_tw():
    name = 'test-from-tw'
    folder = os.path.join(curr_path, 'test')
    path = os.path.join(folder, 'test-from-tw')

    try:
        os.mkdir(folder)

        with tf.Session() as sess:
            model = MockBaseModel([2], 1, [lambda x: tf.Variable(2. , name=name + '/pred')], sess=sess, name=name)
            sess.run(tf.global_variables_initializer())
            model.save(path, sess=sess)

        loaded_model = MockBaseModel.from_tw(path + '.tw', layers=[lambda x: tf.Variable(3., name=name + '/pred')])

        assert loaded_model.name == model.name
        assert loaded_model.X_shape == model.X_shape
        assert loaded_model.y_shape == model.y_shape
    finally:
        remove_dir(folder)
