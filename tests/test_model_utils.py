import pytest
import numpy as np
import tensorflow as tf

from tfwrapper.models.utils import combine_graphs
from tfwrapper.utils.exceptions import IllegalStateException


def test_combine_graphs():
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


def test_combine_graphs_with_session():
    exception = False
    with tf.Session() as sess:
        try:
            graph = combine_graphs(None, None, graph1_out=None, graph2_in=None)
        except IllegalStateException:
            exception = True

    assert exception, 'Calling combine_graphs when a tf.Session is default does not raise an exception'

