import numpy as np
import os
import pandas as pd
import tensorflow as tf

from tfwrapper import logger
from tfwrapper import TFSession
from .pretrained_model import PretrainedModel
from tfwrapper.utils.data import parse_features
from tfwrapper.utils.data import write_features

from .utils import INCEPTION_PB_PATH
from .utils import download_inceptionv3

class InceptionV3(PretrainedModel):
	DEFAULT_INPUT_LAYER = 'Cast:0'
	DEFAULT_FEATURES_LAYER = 'pool_3:0'

	core_layer_names = [
		'DecodeJpeg/contents:0',
		'Cast:0',
		'conv:0',
		'conv_1:0',
		'conv_2:0',
		'pool:0',
		'conv_3:0',
		'conv_4:0',
		'pool_1:0',
		'mixed/join:0',
		'mixed_1/join:0',
		'mixed_2/join:0',
		'mixed_3/join:0',
		'mixed_4/join:0',
		'mixed_5/join:0',
		'mixed_6/join:0',
		'mixed_7/join:0',
		'mixed_8/join:0',
		'mixed_9/join:0',
		'mixed_10/join:0',
		'pool_3:0',
		'softmax:0'
	]

	def __init__(self,graph_file=INCEPTION_PB_PATH):
		graph_file = download_inceptionv3(graph_file, verbose=True)

		if not os.path.isfile(graph_file):
			raise Exception('Invalid path to inception v3 pb file')

		self.graph_file = graph_file

		with tf.gfile.FastGFile(graph_file,'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def,name='')
			
			super().__init__(tf.get_default_graph())

	def run_op(self, to_layer, from_layer, data, sess=None):
		with TFSession(sess, self.graph) as sess:
			logger.info('Extracting features from layer ' + from_layer + ' to ' + to_layer)
			to_tensor = self.graph.get_tensor_by_name(to_layer)
			
			try:
				feature = sess.run(to_tensor,{from_layer: data})
				return feature[0]
			except Exception as e:
				logger.warning('Unable to extract feature')
				logger.warning(str(e))
				raise e

	def extract_features(self, img, layer=DEFAULT_FEATURES_LAYER, sess=None):
		with TFSession(sess, self.graph) as sess:
			return self.run_op(layer, self.DEFAULT_INPUT_LAYER, img, sess=sess)

	def extract_features_from_file(self, filename, layer=DEFAULT_FEATURES_LAYER, sess=None):
		with TFSession(sess, self.graph) as sess:
			try:
				image_data = tf.gfile.FastGFile(filename, 'rb').read()
				feature = self.run_op(layer, 'DecodeJpeg/contents:0', image_data, sess=sess)
				
				return feature
			except Exception as e:
				logger.warning('Unable to read image %' % filename)
				logger.warning(str(e))

				return None
