import tensorflow as tf

from tfwrapper.nets import VGG16
from tfwrapper.utils import get_variable_by_name

from .utils import vgg16_ckpt_path

class PretrainedVGG16(VGG16):
	def __init__(self, X_shape, *, ckpt_path=vgg16_ckpt_path(), sess=None, graph=None, name='vgg_16'):
		super().__init__(X_shape, sess=sess, graph=graph, name='vgg_16')
		self.load_from_checkpoint(ckpt_path, sess=sess)

	def load_from_checkpoint(self, ckpt_path, sess=None):
		variables = [
			'vgg_16/fc7/weights',
			'vgg_16/fc6/weights',
			'vgg_16/conv5/conv5_3/biases',
			'vgg_16/fc8/weights',
			'vgg_16/conv5/conv5_2/weights',
			'vgg_16/conv5/conv5_2/biases',
			'vgg_16/conv5/conv5_1/biases',
			'vgg_16/conv4/conv4_2/weights',
			'vgg_16/conv5/conv5_1/weights',
			'vgg_16/conv4/conv4_2/biases',
			'vgg_16/conv3/conv3_3/biases',
			'vgg_16/fc7/biases',
			'vgg_16/conv3/conv3_2/weights',
			'vgg_16/conv4/conv4_3/biases',
			'vgg_16/conv2/conv2_2/biases',
			'vgg_16/conv3/conv3_2/biases',
			'vgg_16/conv2/conv2_1/weights',
			'vgg_16/conv3/conv3_3/weights',
			'vgg_16/conv1/conv1_1/biases',
			'vgg_16/conv2/conv2_2/weights',
			'vgg_16/conv2/conv2_1/biases',
			'vgg_16/conv4/conv4_1/weights',
			'vgg_16/fc8/biases',
			'vgg_16/fc6/biases',
			'vgg_16/conv3/conv3_1/weights',
			'vgg_16/conv1/conv1_2/weights',
			'vgg_16/conv4/conv4_1/biases',
			'vgg_16/conv5/conv5_3/weights',
			'vgg_16/conv1/conv1_2/biases',
			'vgg_16/conv4/conv4_3/weights',
			'vgg_16/conv1/conv1_1/weights',
			'vgg_16/conv3/conv3_1/biases',
		]

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver([get_variable_by_name(n) for n in variables])
		saver.restore(sess, ckpt_path)
