import os
import json
import numpy as np
import tensorflow as tf

from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.supervisedmodel import METAFILE_SUFFIX
from tfwrapper.utils.exceptions import InvalidArgumentException

from utils import curr_path
from utils import remove_dir

def test_mismatching_lengths():
    model = SingleLayerNeuralNet([28, 28, 1], 3, 5)
    X = np.zeros([50, 28, 28, 1])
    y = np.zeros([100, 3])

    exception = False
    try:
        model.train(X, y, epochs=1)
    except InvalidArgumentException:
        exception = True

    assert exception

def test_invalid_X_shape():
    model = SingleLayerNeuralNet([28, 28, 1], 3, 5)
    X = np.zeros([100, 28, 28, 2])
    y = np.zeros([100, 3])

    exception = False
    try:
        model.train(X, y, epochs=1)
    except InvalidArgumentException:
        exception = True

    assert exception

def test_y_without_onehot():
    model = SingleLayerNeuralNet([28, 28, 1], 3, 5)
    X = np.zeros([100, 28, 28, 1])
    y = np.zeros([100])

    exception = False
    try:
        model.train(X, y, epochs=1)
    except InvalidArgumentException:
        exception = True

    assert exception

def test_invalid_classes():
    model = SingleLayerNeuralNet([28, 28, 1], 3, 5)
    X = np.zeros([100, 28, 28, 1])
    y = np.zeros([100, 5])

    exception = False
    try:
        model.train(X, y, epochs=1)
    except InvalidArgumentException:
        exception = True

    assert exception

def test_save_metadata():
    name = 'Name'
    X_shape = [1, 2, 3]
    y_size = 4
    batch_size = 5

    folder = os.path.join(curr_path, 'test')
    try:
        os.mkdir(folder)
        filename = os.path.join(folder, 'test')

        with tf.Session() as sess:
            model = SingleLayerNeuralNet(X_shape, y_size, 1, name=name, sess=sess)
            model.batch_size = batch_size
            sess.run(tf.global_variables_initializer())
            model.save(filename, sess=sess)

            metadata_filename = '%s.%s' % (filename, METAFILE_SUFFIX)

            assert os.path.isfile(metadata_filename)
            with open(metadata_filename, 'r') as f:
                obj = json.load(f)

            assert name == obj['name']
            assert X_shape == obj['X_shape']
            assert y_size == obj['y_size']
            assert batch_size == obj['batch_size']
    finally:
        remove_dir(folder)

def test_save_labels():
    labels = ['a', 'b', 'c']

    folder = os.path.join(curr_path, 'test')
    try:
        os.mkdir(folder)
        filename = os.path.join(folder, 'test')

        with tf.Session() as sess:
            model = SingleLayerNeuralNet([1], 1, 1, sess=sess)
            sess.run(tf.global_variables_initializer())
            model.save(filename, labels=labels, sess=sess)

            metadata_filename = '%s.%s' % (filename, METAFILE_SUFFIX)

            assert os.path.isfile(metadata_filename)
            with open(metadata_filename, 'r') as f:
                obj = json.load(f)

            assert labels == obj['labels']
    finally:
        remove_dir(folder)

def test_train_X_y():
    model = SingleLayerNeuralNet([1], 1, 5)
    X = np.reshape(np.arange(10), [10, 1])
    y = np.reshape(np.arange(10), [10, 1])

    exception = False
    #try:
    model.train(X, y, epochs=1)
    #except Exception as e:
    #    print(e)
    #    exception = True

    assert not exception

def data_generator(size=10, batch_size=5):
    X = np.reshape(np.arange(size), [size, 1])
    y = np.reshape(np.arange(size), [size, 1])

    for i in range(0, size, batch_size):
        yield X[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]

def test_train_generator():
    model = SingleLayerNeuralNet([1], 1, 5)
    generator = data_generator()

    exception = False
    try:
        model.train(generator=generator, epochs=1)
    except Exception as e:
        print(e)
        exception = True

    assert not exception

def test_no_data():
    model = SingleLayerNeuralNet([10], 3, 5)

    exception = False
    try:
        model.train(epochs=1)
    except Exception:
        exception = True

    assert exception

