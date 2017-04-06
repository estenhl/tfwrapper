import os
import numpy as np
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.utils.data import parse_features
from tfwrapper.utils.data import write_features

INCEPTION_PB_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','inception_v3.pb')

class InceptionV3():
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

	def extract_features_from_img(self, img, layer='pool_3:0', sess=None):
		with TFSession(sess, self.graph) as sess:
			try:
				feature = self.run_op(layer, 'Cast:0', img, sess=sess)
				
				return feature.flatten()
			except Exception as e:
				print(e)

				return None

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
		all_features = []

		with TFSession(sess,self.graph) as sess:
			for filename in os.listdir(folder):
				if filename not in skip:
					src = os.path.join(folder,filename)
					features = self.extract_features_from_file(src,layer=layer,sess=sess)
					if features is not None:
						m = {'filename': filename,'features': features}
						if label is not None:
							m['label'] = label
						all_features.append(m)
				else:
					print('Skipping ' + filename)

		return all_features

	def extract_features_from_datastructure(self,root,skip=[],feature_file=None,layer='pool_3:0',sess=None):
		all_features = []

		if feature_file:
			all_features = parse_features(feature_file)
			skip += [x['filename'] for x in all_features]

		with TFSession(sess, self.graph) as sess:
			for foldername in os.listdir(root):
				folder = os.path.join(root,foldername)
				
				if os.path.isdir(folder):
					all_features += self.extract_features_from_folder(folder,label=foldername,skip=skip,layer=layer,sess=sess)

		if feature_file:
			write_features(feature_file,all_features)

		return all_features
