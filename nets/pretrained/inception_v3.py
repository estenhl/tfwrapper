import os
import tensorflow as tf

class InceptionV3():
	def __init__(self, graph_file=os.path.join('models', 'inception_v3.pb')):
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
		if not sess:
			sess = tf.Session(graph=self.graph)

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

		if not sess:
			sess = tf.Session(self.graph)

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

	def extract_features_from_datastructure(self, root, skip=[], layer='pool_3:0', sess=None):
		all_features = []

		if not sess:
			sess = tf.Session(graph=self.graph)

		for foldername in os.listdir(root):
			folder = os.path.join(root, foldername)
			
			if os.path.isdir(folder):
				all_features += self.extract_features_from_folder(folder, label=foldername, skip=skip, layer=layer, sess=sess)


		return all_features
