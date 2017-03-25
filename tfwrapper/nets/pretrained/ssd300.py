import tensorflow as tf

from .utils import ssd300_ckpt_path

class SSD300:
	def __init__(self, ckpt_file=ssd300_ckpt_path(verbose=True)):
		with tf.Session() as sess:
			saver = tf.train.Saver()
			saver.restore(sess, ckpt_file)
		