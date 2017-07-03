import pytest
import tensorflow as tf

from tfwrapper.layers.loss import mse
from tfwrapper.layers.loss import binary_hinge
from tfwrapper.layers.loss import squared_binary_hinge
from tfwrapper.layers.loss import multiclass_hinge
from tfwrapper.layers.loss import squared_multiclass_hinge


def test_mse():
    name = 'test-mse'

    with tf.Session() as sess:
        y = tf.Variable([0.0, 0.0, 0.0, 0.0])
        preds = tf.Variable([1.0, 2.0, 3.0, 4.0])
        expected_loss = (1**2 + 2**2 + 3**2 + 4**2) / 4

        sess.run(tf.global_variables_initializer())
        loss_tensor = mse(y, preds, name=name)
        result = sess.run(loss_tensor)

    assert expected_loss == result
    assert name + ':0' == loss_tensor.name


def test_binary_hinge():
    name = 'test-binary-hinge'

    with tf.Session() as sess:
        y = tf.Variable([1.0, 0.0, 1.0, 0.0])
        preds = tf.Variable([1.0, 1.5, -2.0, -2.5])
        expected_loss = (0 + 2.5 + 3.0 + 0) / 4

        sess.run(tf.global_variables_initializer())
        loss_tensor = binary_hinge(y, preds, name=name)
        result = sess.run(loss_tensor)

    assert expected_loss == result
    assert name + ':0' == loss_tensor.name


def test_binary_squared_hinge():
    name = 'test-binary-hinge-squared'

    with tf.Session() as sess:
        y = tf.Variable([1.0, 0.0, 1.0, 0.0])
        preds = tf.Variable([1.0, 1.5, -2.0, -2.5])
        expected_loss = ((0 + 2.5 + 3.0 + 0) / 4)**2

        sess.run(tf.global_variables_initializer())
        loss_tensor = squared_binary_hinge(y, preds, name=name)
        result = sess.run(loss_tensor)

    assert expected_loss == result
    assert name + ':0' == loss_tensor.name


def test_hinge():
    name = 'test-hinge'

    with tf.Session() as sess:
        y = [
            [0.0, 0.0, 0.0, 1.0], 
            [0.0, 1.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0, 0.0]
        ]
        var = tf.Variable(y)

        placeholder = tf.placeholder(tf.float32, [None, 4])

        preds = tf.Variable([
            [1.0, 1.5, 0.5, 2.0], 
            [1.5, 0.5, 0.0, 2.0], 
            [1.5, 2.0, 0.5, 1.0], 
            [1.0, 1.5, 2.0, 0.5]])

        expected_loss = ((1 + (1.5 - 2)) + (1 + (2 - 0.5)) + (1 + (2 - 1.5)) + (1 + (1.5 - 2))) / 4

        sess.run(tf.global_variables_initializer())
        loss_tensor = multiclass_hinge(placeholder, preds, name=name)
        result = sess.run(loss_tensor, feed_dict={placeholder: y})

    assert abs(expected_loss - result) < 10e5
    assert name + ':0' == loss_tensor.name


def test_squared_multiclass_hinge():
    name = 'test-hinge-squared'

    with tf.Session() as sess:
        y = [
            [0.0, 0.0, 0.0, 1.0], 
            [0.0, 1.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0, 0.0]
        ]
        var = tf.Variable(y)

        placeholder = tf.placeholder(tf.float32, [None, 4])

        preds = tf.Variable([
            [1.0, 1.5, 0.5, 2.0], 
            [1.5, 0.5, 0.0, 2.0], 
            [1.5, 2.0, 0.5, 1.0], 
            [1.0, 1.5, 2.0, 0.5]])

        expected_loss = (((1 + (1.5 - 2)) + (1 + (2 - 0.5)) + (1 + (2 - 1.5)) + (1 + (1.5 - 2))) / 4) ** 2

        sess.run(tf.global_variables_initializer())
        loss_tensor = multiclass_hinge(placeholder, preds, name=name)
        result = sess.run(loss_tensor, feed_dict={placeholder: y})

    assert abs(expected_loss - result) < 10e5
    assert name + ':0' == loss_tensor.name
