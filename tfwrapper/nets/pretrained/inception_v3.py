import numpy as np
import os
import pandas as pd
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.nets.pretrained.pretrained_model import PretrainedModel
from tfwrapper.utils.data import parse_features
from tfwrapper.utils.data import write_features

from .utils import INCEPTION_PB_PATH
from .utils import download_inceptionv3

class InceptionV3(PretrainedModel):
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

	FEATURE_LAYER = 'pool_3:0'

	def __init__(self,graph_file=INCEPTION_PB_PATH):
		graph_file = download_inceptionv3(graph_file, verbose=True)

		if not os.path.isfile(graph_file):
			raise Exception('Invalid path to inception v3 pb file')

		self.graph_file = graph_file

		with tf.gfile.FastGFile(graph_file,'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def,name='')
			self.graph = tf.get_default_graph()

			tf.reset_default_graph()

	def run_op(self, to_layer, from_layer, data, sess=None):
		print('Extracting features from layer ' + from_layer + ' to ' + to_layer)
		to_tensor = self.graph.get_tensor_by_name(to_layer)

		if sess is None:
			raise NotImplementedError('Needs a sess')
			
		feature = sess.run(to_tensor,{from_layer: data})

		return feature

	def get_feature(self, img, sess, layer='pool_3:0'):
		return self.extract_features_from_img(img, layer=layer, sess=sess)

	def extract_features_from_img(self, img, layer='pool_3:0', sess=None):
		with TFSession(sess, self.graph) as sess:
			try:
				feature = self.run_op(layer, 'Cast:0', img, sess=sess)

				# Very very dirty
				if layer == 'pool_3:0':
					return feature.flatten()
				else:
					return feature
			except Exception as e:
				print(e)

				return None


	def extract_features_from_imgs(self, imgs, layer='pool_3:0', sess=None):
		with TFSession(sess, self.graph) as sess:
			return imgs.apply(lambda x: self.extract_features_from_img(x, layer, sess))


	def extract_features_from_file(self, filename, layer='pool_3:0', sess=None):
		with TFSession(sess, self.graph) as sess:
			try:
				print('Extracting features for ' + filename + ' from layer ' + layer)
				image_data = tf.gfile.FastGFile(filename, 'rb').read()
				feature = self.run_op(layer, 'DecodeJpeg/contents:0', image_data, sess=sess)
				
				return feature.flatten()
			except Exception as e:
				print(e)
				print('Unable to get feature for ' + str(filename))

				return None


	def extract_features_from_files(self,filenames,layer='pool_3:0',sess=None):
		features = []
		with TFSession(sess, self.graph) as sess:
			for filename in filenames:
				features.append(self.extract_features_from_file(filename,layer=layer,sess=sess))


		return np.asarray(features)

	def extract_features_from_folder(self,folder,label=None,skip=[],layer='pool_3:0',sess=None):

		filenames_to_skip = set(os.listdir(folder)) & set(skip)
		filenames_to_include = set(os.listdir(folder)) - set(skip)
		all_features = pd.DataFrame(list(filenames_to_include), columns=['filename'])
		for filename in filenames_to_skip:
			print('Skipping ' + filename)
		all_features['label'] = label
		with TFSession(sess, self.graph) as sess:
			all_features['features'] = (all_features['filename']
										.apply(lambda x: self.extract_features_from_file(os.path.join(folder,x)
																						 ,layer=layer
																						 ,sess=sess))
										)

		return all_features

	def extract_features_from_datastructure(self,root,skip=[],feature_file=None,layer='pool_3:0',sess=None):
		all_features = pd.DataFrame()

		if feature_file:
			all_features = parse_features(feature_file)
			skip += all_features['filename'].tolist()

		with TFSession(sess, self.graph) as sess:
			for foldername in os.listdir(root):
				folder = os.path.join(root,foldername)
				
				if os.path.isdir(folder):
					all_features = all_features.append(self.extract_features_from_folder(folder,label=foldername,skip=skip,layer=layer,sess=sess))

		if feature_file:
			write_features(feature_file,all_features)

		return all_features
