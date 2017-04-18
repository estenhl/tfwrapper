import os
import numpy as np
import tensorflow as tf

from tfwrapper import Dataset
from tfwrapper.datasets import flowers
from tfwrapper.nets.pretrained import InceptionV3
from tfwrapper.dimensionality_reduction import LDA

from utils import curr_path

dataset = flowers()
feature_file = os.path.join(curr_path, 'data', 'flower_features.csv')
features = []
if os.path.isfile(feature_file):
	with open(feature_file, 'r') as f:
		for line in f.readlines():
			features.append([float(x) for x in line.split(',')])
else:
	try:
		with tf.Session() as sess:
			inception = InceptionV3()
			i = 0
			for x in dataset.X:
				features.append(inception.extract_features_from_img(x, sess=sess))
	except Exception:
		pass
	with open(feature_file, 'w') as f:
		for feature in features:
			f.write(','.join([str(x) for x in feature]) + '\r\n')

dataset = Dataset(X=np.asarray(features), y=dataset.y)
dataset = dataset.translate_labels()
dataset = dataset.onehot()
lda = LDA(50)
lda.fit(dataset.X, dataset.y)