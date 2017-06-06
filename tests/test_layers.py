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
"""
def test_batch_normalization():
    X = np.reshape(np.random.uniform(low=0, high=255, size=3 * 3 * 3 * 2), (3, 3, 3, 2))
    var = tf.Variable(X, dtype=tf.float32)
    name = 'test_bn'

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tensor = batch_normalization(X=var, name=name)
        sess.run(tf.global_variables_initializer())
        result = sess.run(tensor)

    mean = np.mean(X, axis=0)
    variance = np.var(X, axis=0)
    for i in range(len(X)):
            print(X[i])
            X[i] = (X[i] - mean) / variance

    # TODO (02.06.17): Reimplement when TF fixes (currently returns name + '/add')
    #assert name + ':0' == tensor.name
    assert X.shape == result.shape
    assert np.array_equal(X, result)

"""
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

def test_channel_means():
    name = 'test_channel_means'
    imgs = np.zeros((2, 4, 4, 3))
    for i in range(2):
        for j in range(4):
            for k in range(4):
                imgs[i][j][k] = np.asarray([1, 2, 3]) * (i + 1)

    layer = channel_means(means=np.asarray([1, 2, 3]))

    with tf.Session() as sess:
        tensor = layer(tf.Variable(imgs, dtype=tf.float32))
        sess.run(tf.global_variables_initializer())
        result = sess.run(tensor)

    for i in range(2):
        for j in range(4):
            for k in range(4):
                assert np.array_equal(result[i][j][k], np.asarray([1, 2, 3]) * i)

