import os
import tensorflow as tf

from tfwrapper import ImageDataset
from tfwrapper import FeatureLoader
from tfwrapper import ImagePreprocessor
from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.nets.pretrained import Inception_v4

curr_path = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
data_path = os.path.join(curr_path, '..', 'data', 'datasets', 'catsdogs', 'images')

dataset = ImageDataset(root_folder=data_path)
dataset = dataset.balance(max=20)
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

inc_v4 = Inception_v4()
feature_file = os.path.join(curr_path, 'data', 'catsdogs_features.csv')

train_prep = ImagePreprocessor()
train_prep.resize_to = (299, 299)
train_prep.flip_lr = True
train_loader = FeatureLoader(inc_v4, feature_file=feature_file, preprocessor=train_prep)
train.loader = train_loader

test_prep = ImagePreprocessor()
test_prep.resize_to = (299, 299)
test_loader = FeatureLoader(inc_v4, feature_file=feature_file, preprocessor=test_prep)
test.loader = test_loader

with tf.Session() as sess:
    nn = SingleLayerNeuralNet([1536], 2, 1024, sess=sess, name='InceptionV4Test')
    nn.train(train.X, train.y, epochs=10, sess=sess, verbose=True)
    _, acc = nn.validate(test.X, test.y, sess=sess)
    print('Acc: %d%%' % (acc * 100))

