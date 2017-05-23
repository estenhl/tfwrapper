import pytest
import numpy as np
import tensorflow as tf

from tfwrapper.layers import *

def test_bias():
    size = 5
    name = 'test_bias'

    tensor = bias(size, name=name)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(tensor)

    assert size == len(result)
    assert name + ':0' == tensor.name

def test_weight():
    shape = [5]
    name = 'test_weight'

    tensor = weight(shape, name=name)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(tensor)
        evaluated_shape = sess.run(tf.shape(tensor))

    assert shape == evaluated_shape
    assert name + ':0' == tensor.name

def test_reshape():
    shape = np.asarray([10, 10])
    length = np.prod(shape)
    layer = reshape(shape)

    values = np.zeros(length)
    with tf.Session() as sess:
        result = sess.run(layer(values))

    assert np.array_equal(shape, result.shape)

def test_relu():
    name = 'test_relu'
    values = np.zeros([10, 10])
    layer = relu(name=name)

    with tf.Session() as sess:
        tensor = layer(values)
        result = sess.run(tensor)

    assert name + ':0' == tensor.name
    assert np.array_equal(values.shape, result.shape)

def test_softmax():
    name = 'test_softmax'
    values = np.zeros([10, 10])
    layer = softmax(name=name)

    with tf.Session() as sess:
        tensor = layer(values)
        result = sess.run(tensor)

    assert name + ':0' == tensor.name
    assert np.array_equal(values.shape, result.shape)

def test_fullyconnected():
    name = 'test_fullyconnected'
    shape = np.asarray([10, 10, 10])
    values = np.ones(shape).astype(np.float32)
    inputs = int(np.prod(shape[1:]))
    outputs = 5
    layer = fullyconnected(inputs=inputs, outputs=outputs, name=name)

    with tf.Session() as sess:
        tensor = layer(values)
        sess.run(tf.global_variables_initializer())
        result = sess.run(tensor)

    assert name + ':0' == tensor.name
    assert 10 == len(result)
    assert outputs == len(result[0])

def test_dropout():
    name = 'test_dropout'
    values = np.zeros([10, 10])
    layer = dropout(0.5, name=name)

    with tf.Session() as sess:
        tensor = layer(values)
        result = sess.run(tensor)

    assert name + '/mul:0' == tensor.name
    assert np.array_equal(values.shape, result.shape)