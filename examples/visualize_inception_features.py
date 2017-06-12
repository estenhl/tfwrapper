import os
import numpy as np
import tensorflow as tf

from tfwrapper import Dataset
from tfwrapper import FeatureLoader
from tfwrapper.datasets import flowers
from tfwrapper.models.frozen import FrozenInceptionV3
from tfwrapper.visualization import plot_clusters
from tfwrapper.dimensionality_reduction import PCA

from utils import curr_path

dataset = flowers(size=500)
features_file = os.path.join(curr_path, 'data', 'flowers_inceptionv3.csv')

inception = FrozenInceptionV3()
loader = FeatureLoader(inception, cache=features_file)
dataset.loader = loader

pca = PCA(50, name='VisualizeFlowersExample')
pca.fit(dataset.X)
X = pca.transform(dataset.X)

from sklearn.manifold import TSNE

model = TSNE(n_components=2, perplexity=30.0, early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000, n_iter_without_progress=30, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5)
X = model.fit_transform(X)
dataset = Dataset(X=X, y=dataset.y)
clusters, labels = dataset.get_clusters(get_labels=True)
plot_clusters(clusters, names=labels)
