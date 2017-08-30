import pytest
import tensorflow as tf

from tfwrapper import TFSession


def test_variable_assignment():
    with tf.Session() as sess:
        var = tf.Variable([2.])
        graph = sess.graph

    value = [3.]
    variables = {'test': {'tensor': var, 'value': value}}

    with TFSession(graph=graph, variables=variables) as sess:
        num_ops = len(sess.graph.get_operations())
    
    with TFSession(graph=graph, variables=variables) as sess:
        assert num_ops == len(sess.graph.get_operations())
        assert value == sess.run(var)