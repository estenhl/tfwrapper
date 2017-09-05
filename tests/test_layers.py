import pytest
import numpy as np

from tfwrapper.layers import bias, weight, reshape, relu, softmax, fullyconnected, dropout, channel_means, concatenate, zoom, unet_block, deconv2d

from fixtures import tf

def test_bias(tf):
    size = 5
    name = 'test_bias'

    tensor = bias(size, name=name)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(tensor)

    assert size == len(result)
    assert name + ':0' == tensor.name


def test_weight(tf):
    shape = [5]
    name = 'test_weight'

    tensor = weight(shape, name=name)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(tensor)
        evaluated_shape = sess.run(tf.shape(tensor))

    assert shape == evaluated_shape
    assert name + ':0' == tensor.name



def test_reshape(tf):
    shape = np.asarray([10, 10])
    length = np.prod(shape)
    layer = reshape(shape=shape)

    values = np.zeros(length)
    with tf.Session() as sess:
        result = sess.run(layer(values))

    assert np.array_equal(shape, result.shape)


def test_relu(tf):
    name = 'test_relu'
    values = np.zeros([10, 10])
    layer = relu(name=name)

    with tf.Session() as sess:
        tensor = layer(values)
        result = sess.run(tensor)

    assert name + ':0' == tensor.name
    assert np.array_equal(values.shape, result.shape)


def test_softmax(tf):
    name = 'test_softmax'
    values = np.zeros([10, 10])
    layer = softmax(name=name)

    with tf.Session() as sess:
        tensor = layer(values)
        result = sess.run(tensor)

    assert name + ':0' == tensor.name
    assert np.array_equal(values.shape, result.shape)


def test_fullyconnected(tf):
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


def test_dropout(tf):
    name = 'test_dropout'
    values = np.zeros([10, 10])
    layer = dropout(keep_prob=0.5, name=name)

    with tf.Session() as sess:
        tensor = layer(values)
        result = sess.run(tensor)

    # Reinstate when TF fixed naming convention
    #assert name + ':0' == tensor.name
    assert np.array_equal(values.shape, result.shape)


def test_channel_means(tf):
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


def test_concatenate(tf):
    name = 'test_concatenate'

    zeros = np.zeros((3, 3, 3, 3))
    ones = np.ones((3, 3, 3, 3))

    with tf.Session() as sess:
        x1 = tf.Variable(zeros)
        x2 = tf.Variable(ones)
        sess.run(tf.global_variables_initializer())

        tensor = concatenate([x1, x2], name=name)
        result = sess.run(tensor)
        expected_result = np.concatenate([zeros, ones], axis=3)

        assert np.array_equal(expected_result, result), 'Concatenate does not use axis len(shape)-1 as default'
        assert name + ':0' == tensor.name, 'Concatenate tensor does not get correct name'

        for i in range(4):
            result = sess.run(concatenate([x1, x2], axis=i, name='%s_%d' % (name, i)))
            expected_result = np.concatenate([zeros, ones], axis=i)

            assert np.array_equal(expected_result, result), 'Concatenate does not work for axis %d' % i


def test_zoom(tf):
    name = 'test_zoom'

    data = np.zeros((1, 5, 5, 3)).astype(float)

    for i in range(5):
        for j in range(5):
            data[0][i][j] = np.tile((i * 5) + j, 3)

    expected_result = data[:, 1:4, 1:4]
    print(expected_result)

    with tf.Session() as sess:
        X = tf.placeholder(tf.float32, (None, 5, 5, 3))

        tensor = zoom(X, size=(3, 3), name=name)
        result = sess.run(tensor, feed_dict={X: data})

        assert (1, 3, 3, 3) == result.shape, 'Zooming does not fetch window of correct size'
        assert np.array_equal(expected_result, result), 'Zooming does not fetch correct window'
        assert name + ':0' == tensor.name, 'Zooming tensor does not get correct name'


def test_unet_block_shape(tf):
    with tf.Session() as sess:
        X = tf.ones((5, 12, 12, 3))

        tensor = unet_block(X=X, depth=5)
        sess.run(tf.global_variables_initializer())
        result = sess.run(tensor)

    _, height, width, channels = result.shape
    assert 8 == height, 'unet_block does not prune away 4 pixels in height'
    assert 8 == width, 'unet_block does not prune away 4 pixels in width'
    assert 5 == channels, 'unet_block does not yield correct number of channels'


def test_unet_block_name(tf):
    name = 'test_unet_block'

    X = tf.ones((5, 12, 12, 3))
    tensor = unet_block(X=X, depth=3, name=name)

    assert name + ':0' == tensor.name, 'unet_block is not given correct name (%s instead of %s)' % (tensor.name, name + ':0')


def test_deconv2d_shape(tf):
    with tf.Session() as sess:
        X = tf.ones((5, 12, 12, 20))

        tensor = deconv2d(X, filter=[2, 2], depth=10)
        sess.run(tf.global_variables_initializer())
        result = sess.run(tensor)
        print('TEST_DECONV2d_SHAPE:')
        print(len(sess.graph.get_operations()))
        print(sess.graph.get_operations())

    assert (5, 24, 24, 10) == result.shape


def test_deconv2d_name(tf):
    name = 'test_deconv2d'

    X = tf.ones((5, 12, 12, 3))
    tensor = deconv2d(X=X, filter=[2, 2], depth=10, name=name)

    assert name + ':0' == tensor.name, 'deconv2d is not given correct name (%s instead of %s)' % (tensor.name, name + ':0')

