import numpy as np
import tensorflow as tf

from tfwrapper.utils.exceptions import InvalidArgumentException

def kullback_leibler(P, Q, name='KullbackLeibler', sess=None):
	if sess.run(tf.rank(P)) == 2 and sess.run(tf.rank(Q)) == 2:
		return kullback_leibler_2d(P, Q, name=name, sess=sess)
	else:
		raise NotImplementedError('Kullback Leibler is only implemented for 2d matrices')

def kullback_leibler_2d(P, Q, name='KullbackLeibler2d', sess=None):
	if not (sess.run(tf.rank(P)) == 2 and sess.run(tf.rank(Q)) == 2):
		raise InvalidArgumentException('Kullback Leibler 2d requires two 2d matrices')

	length = sess.run(tf.shape(P))[0]
	if not length == sess.run(tf.shape(P))[1] == sess.run(tf.shape(Q))[0] == sess.run(tf.shape(Q))[1]:
		raise InvalidArgumentException('Kullback Leibler 2d requires two square matrices with equal dimensions')


	if not P.dtype == tf.float32:
		P = tf.cast(P, tf.float32)
	if not Q.dtype == tf.float32:
		Q = tf.cast(Q, tf.float32)

	# Zero out identity horizontal
	identity = tf.Variable(initial_value=np.identity(length), dtype=tf.float32, name=name + '/identity')
	ones = tf.ones(shape=[length, length], dtype=tf.float32, name=name + '/ones')
	sess.run(tf.variables_initializer([identity]))
	reverse_identity = tf.subtract(ones, identity, name=name + '/reverse_identity')
	transformed_P = P * reverse_identity

	transformed_P = transformed_P * tf.log(P / Q)

	return tf.reduce_sum(transformed_P, name=name)

