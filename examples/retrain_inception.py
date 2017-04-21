import tensorflow as tf
from tfwrapper.containers import CachedFeatureLoader
from tfwrapper.containers import ImageContainer
from tfwrapper.containers import ImageLoader
from tfwrapper.containers import ImagePreprocess
from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.nets.pretrained.inception_v4 import Inception_v4
from tfwrapper.nets.pretrained.inception_v4 import Inception_v4
from tfwrapper.datasets import catsdogs

import os

feature_cache = os.path.join(catsdogs.FILE_PATH, "inc4_cache.csv")

catsdogs.download_cats_and_dogs()

container = catsdogs.create_container(max_images=20)
container.shuffle()
container.one_hot_encode()

container = container.balance(max_value=10)  # example for balancing

train, test = container.split(shape=[0.9, 0.1])

img_size = (299, 299)

train_preprocess = ImagePreprocess()
train_preprocess.resize(img_size=img_size)
train_preprocess.append_flip_lr()

test_preprocess = ImagePreprocess()
test_preprocess.resize(img_size=img_size)

inc_v4 = Inception_v4()
#Inception_v3 could also be used
#inc_v3 = Inception_v3()


# loader = ImageLoader()
loader = CachedFeatureLoader(feature_cache, inc_v4,
                             Inception_v4.FEATURE_LAYER)  # Feature layer could be defaulted in model

train_dataset, train_names = loader.create_dataset(train, train_preprocess)

test_dataset, test_names = loader.create_dataset(test, test_preprocess)

graph = tf.Graph()
with graph.as_default():
    with tf.Session(graph=graph) as sess:
        nn = SingleLayerNeuralNet([train_dataset.X.shape[1]], 2, 1024, sess=sess, name='InceptionV4Test')
        nn.train(train_dataset.X, train_dataset.y, epochs=10, sess=sess, verbose=True)
        _, acc = nn.validate(test_dataset.X, test_dataset.y, sess=sess)
        # nn.save(model_path, sess=sess)
