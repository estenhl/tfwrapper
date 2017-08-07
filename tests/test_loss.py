import pytest
import numpy as np
import tensorflow as tf

from tfwrapper.layers.loss import *


def test_mse():
    name = 'test-mse'

    with tf.Session() as sess:
        y = tf.Variable([0.0, 0.0, 0.0, 0.0])
        preds = tf.Variable([1.0, 2.0, 3.0, 4.0])
        expected_loss = (1**2 + 2**2 + 3**2 + 4**2) / 4

        sess.run(tf.global_variables_initializer())
        loss_tensor = mse(y=y, preds=preds, name=name)
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
        loss_tensor = binary_hinge(y=y, preds=preds, name=name)
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
        loss_tensor = squared_binary_hinge(y=y, preds=preds, name=name)
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
        loss_tensor = multiclass_hinge(y=placeholder, preds=preds, name=name)
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
        loss_tensor = multiclass_hinge(y=placeholder, preds=preds, name=name)
        result = sess.run(loss_tensor, feed_dict={placeholder: y})

    assert abs(expected_loss - result) < 10e5
    assert name + ':0' == loss_tensor.name


def test_pixelwise_softmax_cross_entropy():
    name = 'test-pixelwise-softmax-cross-entropy'
    y = np.asarray([
            [
                [
                    [0., 1., 0.],
                    [1., 0., 0.],
                    [0., 1., 0.]
                ],[
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]
                ]
            ]
        ])

    preds = np.asarray([
            [
                [
                    [1., 3., 1.],
                    [1., 3., 1.],
                    [1., 3., 1.]
                ],[
                    [1., 3., 1.],
                    [1., 3., 1.],
                    [1., 3., 1.]
                ],

            ]
        ])
    

    with tf.Session() as sess:
        y_placeholder = tf.placeholder(tf.float32, [None, 2, 3, 3])
        preds_placeholder = tf.placeholder(tf.float32, [None, 2, 3, 3])

        loss_tensor = pixelwise_softmax_cross_entropy(y=y, preds=preds, name=name)
        result = sess.run(loss_tensor, feed_dict={y_placeholder: y, preds_placeholder: preds})

    expected_result = -(np.log(.6) + np.log(.2) + np.log(.6) + np.log(.2) + np.log(.6) + np.log(.2))/6

    assert abs(expected_result - result) < 1


def test_mean_softmax_cross_entropy():
    name = 'test-mean-softmax-cross-entropy'

    with tf.Session() as sess:
        y = tf.Variable([
            [0.0, 0.0, 0.0, 1.0], 
            [0.0, 1.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0, 0.0]
        ])

        placeholder = tf.placeholder(tf.float32, [None, 4])

        preds = tf.Variable([
            [1.0, 1.5, 0.5, 2.0], 
            [1.5, 0.5, 0.0, 2.0], 
            [1.5, 2.0, 0.5, 1.0], 
            [1.0, 1.5, 2.0, 0.5]
        ])

        expected_loss = -((1 - np.log(0.5)) - (1 - np.log(0.125)) - (1 - np.log(0.375)) - (1 - np.log(0.5)))/4
        tensor = mean_softmax_cross_entropy(y=y, preds=preds, name=name)
        sess.run(tf.global_variables_initializer())
        loss = sess.run(tensor)

    assert name + ':0' == tensor.name
    assert abs(loss - expected_loss) < 1e-2


def test_from_name():
    loss = Loss.from_name('MSE')

    assert loss is not None


def test_from_name_invalid():
    exception = False

    try:
        loss = Loss.from_name('Invalid')
    except NotImplementedError:
        exception = True

    assert exception

