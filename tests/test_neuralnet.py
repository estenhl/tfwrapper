import os
import json
import numpy as np

from tfwrapper.models.nets import SingleLayerNeuralNet
from tfwrapper.models.nets import NeuralNet
from tfwrapper.models.nets.neural_net import METADATA_SUFFIX
from tfwrapper.utils.exceptions import InvalidArgumentException

from fixtures import tf
from utils import curr_path, remove_dir


def test_mismatching_lengths(tf):
    model = SingleLayerNeuralNet([28, 28, 1], 3, 5)
    X = np.zeros([50, 28, 28, 1])
    y = np.zeros([100, 3])

    exception = False
    try:
        model.train(X, y, epochs=1)
    except InvalidArgumentException:
        exception = True

    assert exception


def test_invalid_X_shape(tf):
    model = SingleLayerNeuralNet([28, 28, 1], 3, 5)
    X = np.zeros([100, 28, 28, 2])
    y = np.zeros([100, 3])

    exception = False
    try:
        model.train(X, y, epochs=1)
    except InvalidArgumentException:
        exception = True

    assert exception


def test_y_without_onehot(tf):
    model = SingleLayerNeuralNet([28, 28, 1], 3, 5)
    X = np.zeros([100, 28, 28, 1])
    y = np.zeros([100])

    exception = False
    try:
        model.train(X, y, epochs=1)
    except InvalidArgumentException:
        exception = True

    assert exception


def test_save_metadata(tf):
    name = 'Name'
    X_shape = [1, 2, 3]
    y_shape = [4]
    batch_size = 5

    folder = os.path.join(curr_path, 'test')
    try:
        os.mkdir(folder)
        filename = os.path.join(folder, 'test')

        with tf.Session() as sess:
            model = SingleLayerNeuralNet(X_shape, y_shape, 1, name=name, sess=sess)
            model.batch_size = batch_size
            sess.run(tf.global_variables_initializer())
            model.save(filename, sess=sess)

            metadata_filename = '%s.%s' % (filename, METADATA_SUFFIX)

            assert os.path.isfile(metadata_filename)
            with open(metadata_filename, 'r') as f:
                obj = json.load(f)

            assert name == obj['name']
            assert X_shape == obj['X_shape']
            assert y_shape == obj['y_shape']
            assert batch_size == obj['batch_size']
    finally:
        remove_dir(folder)


def test_save_labels(tf):
    labels = ['a', 'b', 'c']

    folder = os.path.join(curr_path, 'test')
    try:
        os.mkdir(folder)
        filename = os.path.join(folder, 'test')

        with tf.Session() as sess:
            model = SingleLayerNeuralNet([1], 1, 1, sess=sess)
            sess.run(tf.global_variables_initializer())
            model.save(filename, labels=labels, sess=sess)

            metadata_filename = '%s.%s' % (filename, METADATA_SUFFIX)

            assert os.path.isfile(metadata_filename)
            with open(metadata_filename, 'r') as f:
                obj = json.load(f)

            assert labels == obj['labels']
    finally:
        remove_dir(folder)


def test_train_X_y(tf):
    model = SingleLayerNeuralNet([1], 1, 5)
    X = np.reshape(np.arange(10), [10, 1])
    y = np.reshape(np.arange(10), [10, 1])

    exception = False
    try:
        model.train(X, y, epochs=1)
    except Exception as e:
        print(e)
        exception = True

    assert not exception


def data_generator(size=10, batch_size=5):
    X = np.reshape(np.arange(size), [size, 1])
    y = np.reshape(np.arange(size), [size, 1])

    for i in range(0, size, batch_size):
        yield X[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]


def test_train_generator(tf):
    model = SingleLayerNeuralNet([1], 1, 5)
    generator = data_generator()

    exception = False
    try:
        model.train(generator=generator, epochs=1)
    except Exception as e:
        print(e)
        exception = True

    assert not exception


def test_no_data(tf):
    model = SingleLayerNeuralNet([10], 3, 5)

    exception = False
    try:
        model.train(epochs=1)
    except Exception:
        exception = True

    assert exception


def test_load_from_tw(tf):
    data = np.random.rand(10, 10)
    folder = os.path.join(curr_path, 'test')
    try:
        os.mkdir(folder)
        filename = os.path.join(folder, 'test')

        with tf.Session() as sess:
            model = SingleLayerNeuralNet([10], 3, 5, name='FromTw', sess=sess)
            sess.run(tf.global_variables_initializer())
            preds = model.predict(data, sess=sess)
            model.save(filename, sess=sess)

        tf.reset_default_graph()
        loaded_model = NeuralNet.from_tw(filename)

        assert np.array_equal(preds, loaded_model.predict(data))

    finally:
        remove_dir(folder)


def test_get_tensor_by_name(tf):
    name = 'test-get-tensor-by-name'
    with tf.Session() as sess:
        model = SingleLayerNeuralNet([10], 3, 5, name=name, sess=sess)
        tensor = model.get_tensor(name + '/hidden:0')

    assert tensor is not None


def test_get_tensor_by_id(tf):
    name = 'test-get-tensor-by-id'
    with tf.Session() as sess:
        model = SingleLayerNeuralNet([10], 3, 5, name=name, sess=sess)
        tensor = model.get_tensor(-1)

    assert tensor is not None


def test_run_op(tf):
    name = 'test-run-op'
    num_hidden = 3
    batch_size = 10
    with tf.Session() as sess:
        model = SingleLayerNeuralNet([10], 3, num_hidden, name=name, sess=sess)
        sess.run(tf.global_variables_initializer())
        values = model.run_op(name + '/hidden:0', data=np.zeros((batch_size, 10)), sess=sess)

    assert (batch_size, num_hidden) == values.shape


def test_run_op_with_source(tf):
    name = 'test-run-op-with-source'
    num_hidden = 3
    num_classes = 2
    batch_size = 10
    with tf.Session() as sess:
        model = SingleLayerNeuralNet([10], num_classes, num_hidden, name=name, sess=sess)
        sess.run(tf.global_variables_initializer())
        values = model.run_op(name + '/pred:0', data=np.zeros((batch_size, num_hidden)), source=name + '/hidden:0', sess=sess)

    assert (batch_size, num_classes) == values.shape
