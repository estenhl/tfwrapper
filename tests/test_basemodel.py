import pytest
import tensorflow as tf

from tfwrapper.layers import Layer
from tfwrapper.models.basemodel import _parse_layer_list
from tfwrapper.utils.exceptions import InvalidArgumentException

from mock import MockBaseModel
from utils import softmax_wrapper

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

    assert name == model.name
    
