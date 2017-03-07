import os
import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.utils.data import parse_features
from tfwrapper.utils.data import write_features

INCEPTION_PB_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'inception_v3.pb')

class InceptionV3():
	def __init__(self, graph_file=INCEPTION_PB_PATH):
		if not os.path.isfile(graph_file):
			raise Exception('Invalid path to inception v3 pb file')

		self.graph_file = graph_file

		with tf.gfile.FastGFile(graph_file, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='')
			self.graph = tf.get_default_graph()
			tf.reset_default_graph()

	def extract_features_from_file(self, filename, layer='pool_3:0', sess=None):
		with TFSession(sess, self.graph) as sess:
			tensor = sess.graph.get_tensor_by_name(layer)
			try:
				print('Extracting features for ' + filename)
				image_data = tf.gfile.FastGFile(filename, 'rb').read()
				feature = sess.run(tensor, {'DecodeJpeg/contents:0': image_data})

				return feature.flatten()
			except Exception as e:
				print(e)
				print('Unable to get feature for ' + str(filename))

				return None

	def extract_features_from_folder(self, folder, label=None, skip=[], layer='pool_3:0', sess=None):
		all_features = []

		with TFSession(sess, self.graph) as sess:
			for filename in os.listdir(folder):
				if filename not in skip:
					src = os.path.join(folder, filename)
					features = self.extract_features_from_file(src, layer=layer, sess=sess)
					if features is not None:
						m = {'filename': filename, 'features': features}
						if label is not None:
							m['label'] = label
						all_features.append(m)
				else:
					print('Skipping ' + filename)

		return all_features

	def extract_features_from_datastructure(self, root, skip=[], feature_file=None, layer='pool_3:0', sess=None):
		all_features = []

		if feature_file:
			all_features = parse_features(feature_file)
			skip += [x['filename'] for x in all_features]

		with TFSession(sess, self.graph) as sess:
			for foldername in os.listdir(root):
				folder = os.path.join(root, foldername)
				
				if os.path.isdir(folder):
					all_features += self.extract_features_from_folder(folder, label=foldername, skip=skip, layer=layer, sess=sess)

		if feature_file:
			write_features(feature_file, all_features)

		return all_features
