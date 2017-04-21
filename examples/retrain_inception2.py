import os
import tensorflow as tf

from tfwrapper import ImageDataset
from tfwrapper import FeatureExtractor
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

train.preprocessor = FeatureExtractor(inc_v4, os.path.join(curr_path, 'data', 'catsdogs_features.csv'))
train.preprocessor.resize_to = (299, 299)
train.preprocessor.flip_lr = True

test.preprocessor = FeatureExtractor(inc_v4, os.path.join(curr_path, 'data', 'catsdogs_features.csv'))
test.preprocessor.resize_to = (299, 299)

with tf.Session(graph=inc_v4.graph) as sess:
    train.preprocessor.sess = sess
    test.preprocessor.sess = sess

    nn = SingleLayerNeuralNet([1536], 2, 1024, sess=sess, name='InceptionV4Test')
    nn.train(train.X, train.y, epochs=10, sess=sess, verbose=True)
    _, acc = nn.validate(test.X, test.y, sess=sess)
    print('Acc: %d' % (acc * 100))

