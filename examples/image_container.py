import tensorflow as tf
from tfwrapper.containers import CachedFeatureLoader
from tfwrapper.containers import ImageContainer
from tfwrapper.containers import ImageLoader
from tfwrapper.containers import ImagePreprocess
from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.nets.pretrained.inception_v4 import Inception_v4

dataset_path = "path/to/dataset"
feature_cache = "path/to/feature_cache.csv"
inception_file = "path/to/inception_v4" #can be default value in model
model_path = "path/to/model"


container = ImageContainer(dataset_path)
container.shuffle()
container.one_hot_encode()

container = container.balance(max_value=1000) #should also be possible to choose max_value == min of smallest class

train, test = container.split(shape=[0.85, 0.15])

img_size = (299, 299)

train_preprocess = ImagePreprocess()
train_preprocess.resize(img_size=img_size)
train_preprocess.append_flip_lr()

test_preprocess = ImagePreprocess()
test_preprocess.resize(img_size=img_size)

inc_v4 = Inception_v4(inception_file)

#loader = ImageLoader()
loader = CachedFeatureLoader(feature_cache, inc_v4, Inception_v4.FEATURE_LAYER) #Feature layer could be defaulted in model

generator = loader.batch_generator()
for i in range(100):
    print(next(generator))

train_dataset, train_names = loader.create_dataset(train, train_preprocess)

test_dataset, test_names = loader.create_dataset(test, test_preprocess)

gen = loader.batch_generator(train, train_preprocess, 16, True)




graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		nn = SingleLayerNeuralNet([train_dataset.X.shape[1]], 2, 1024, sess=sess, graph=graph, name='InceptionV3Test')
		nn.train(train_dataset.X, train_dataset.Y, epochs=10, sess=sess, verbose=True)
		nn.save(model_path, sess=sess)


nn = SingleLayerNeuralNet([test_dataset.X.shape[1]], 2, 1024, name='InceptionV3Test')
graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		nn.load(model_path, sess=sess)
		_, acc = nn.validate(test_dataset.X, test_dataset.Y, sess=sess)
		print('Inception Test accuracy: %d %%' % (acc*100))

#Train model