import os
import numpy as np
import tensorflow as tf

from tfwrapper import Dataset
from tfwrapper.datasets import flowers
from tfwrapper.nets.pretrained import InceptionV3
from tfwrapper.visualization import plot_clusters
from tfwrapper.dimensionality_reduction import PCA

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
pca = PCA(50, name='VisualizeFlowersExample')
pca.fit(dataset.X)
X = pca.transform(dataset.X)

from sklearn.manifold import TSNE

model = TSNE(n_components=2, perplexity=30.0, early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000, n_iter_without_progress=30, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5)
print('Doing TSNE')
X = model.fit_transform(X)
dataset = Dataset(X=X, y=dataset.y)
clusters, labels = dataset.get_clusters(get_labels=True)
plot_clusters(clusters, names=labels)
