import pytest
import numpy as np

from tfwrapper.models.utils import combine_graphs
from tfwrapper.utils.exceptions import IllegalStateException

from fixtures import tf


def test_combine_graphs(tf):
    graph1 = tf.Graph()
    with graph1.as_default():
        with tf.Session(graph=graph1) as sess:
            var1 = tf.Variable([1, 1, 1], name='var1')
            tensor1 = tf.multiply(var1, 2, name='tensor1')

    graph2 = tf.Graph()
    with graph2.as_default():
        with tf.Session(graph=graph2) as sess:
            placeholder = tf.placeholder(tf.int32, [3], name='placeholder')
            tensor2 = tf.multiply(placeholder, 2, name='tensor2')


    graph, init_op = combine_graphs(graph1, graph2, graph1_out='tensor1:0', graph2_in='placeholder:0')

    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            sess.run(init_op)
            tensor = graph.get_tensor_by_name('graph2/tensor2:0')
            result = sess.run(tensor)

    assert np.array_equal([4, 4, 4], result)


def test_combine_graphs_with_multiple_variables(tf):
    graph1 = tf.Graph()
    with graph1.as_default():
        with tf.Session(graph=graph1) as sess:
            var1 = tf.Variable([1, 1, 1], name='var1')
            var2 = tf.Variable([1, 1, 1], name='var2')
            add1 = var1 + var2
            tensor1 = tf.multiply(add1, 2, name='tensor1')

    graph2 = tf.Graph()
    with graph2.as_default():
        with tf.Session(graph=graph2) as sess:
            placeholder = tf.placeholder(tf.int32, [3], name='placeholder')
            var3 = tf.Variable([1, 1, 1], name='var3')
            add2 = placeholder + var3
            tensor2 = tf.multiply(add2, 2, name='tensor2')


    graph, init_op = combine_graphs(graph1, graph2, graph1_out='tensor1:0', graph2_in='placeholder:0')

    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            sess.run(init_op)
            tensor = graph.get_tensor_by_name('graph2/tensor2:0')
            result = sess.run(tensor)

    assert np.array_equal([10, 10, 10], result)


def test_combine_graphs_with_placeholders(tf):
    graph1 = tf.Graph()
    with graph1.as_default():
        with tf.Session(graph=graph1) as sess:
            placeholder1 = tf.placeholder(tf.int32, [3], name='placeholder1')
            tensor1 = tf.multiply(placeholder1, 2, name='tensor1')

    graph2 = tf.Graph()
    with graph2.as_default():
        with tf.Session(graph=graph2) as sess:
            placeholder2 = tf.placeholder(tf.int32, [3], name='placeholder2')
            tensor2 = tf.multiply(placeholder2, 2, name='tensor2')


    graph, init_op = combine_graphs(graph1, graph2, graph1_out='tensor1:0', graph2_in='placeholder2:0')

    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            sess.run(init_op)
            tensor = graph.get_tensor_by_name('graph2/tensor2:0')
            placeholder = graph.get_tensor_by_name('graph1/placeholder1:0')
            result = sess.run(tensor, feed_dict={placeholder: [2, 2, 2]})

    assert np.array_equal([8, 8, 8], result)


def test_combine_graphs_with_session(tf):
    exception = False
    with tf.Session() as sess:
        try:
            graph = combine_graphs(None, None, graph1_out=None, graph2_in=None)
        except IllegalStateException:
            exception = True

    assert exception, 'Calling combine_graphs when a tf.Session is default does not raise an exception'


def test_combine_graphs_name_scopes(tf):
    graph1 = tf.Graph()
    with graph1.as_default():
        with tf.Session(graph=graph1) as sess:
            placeholder1 = tf.placeholder(tf.int32, [3], name='placeholder1')
            tensor1 = tf.multiply(placeholder1, 2, name='tensor1')

    graph2 = tf.Graph()
    with graph2.as_default():
        with tf.Session(graph=graph2) as sess:
            placeholder2 = tf.placeholder(tf.int32, [3], name='placeholder2')
            tensor2 = tf.multiply(placeholder2, 2, name='tensor2')


    graph, init_op = combine_graphs(graph1, graph2, graph1_out='tensor1:0', graph2_in='placeholder2:0', graph1_name='test_graph1', graph2_name='test_graph2')

    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            sess.run(init_op)
            placeholder = graph.get_tensor_by_name('test_graph1/placeholder1:0')
            tensor = graph.get_tensor_by_name('test_graph2/tensor2:0')
            result = sess.run(tensor, feed_dict={placeholder: [1, 1, 1]})

    assert np.array_equal([4, 4, 4], result)
