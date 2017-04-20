import os
import tensorflow as tf

from tfwrapper import ImageDataset
from tfwrapper import FeatureExtractor
from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.nets.pretrained import Inception_v4

curr_path = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
data_path = os.path.join(curr_path, '..', 'data', 'datasets', 'catsdogs', 'images')

dataset = ImageDataset(root_folder=data_path)
dataset = dataset[:20]
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.balance(max=5)
dataset = dataset.onehot()
train, test = dataset.split(0.9)

"""
inc_v4 = Inception_v4()

train.preprocessor = FeatureExtractor(inc_v4, 'train_features.tmp')
train.preprocessor.resize_to = (299, 299)
train.preprocessor.flip_lr = True

test.preprocessor = FeatureExtractor(inc_v4, 'test_features.tmp')
test.preprocessor.resize_to = (299, 299)
"""
print(train.X.shape)
exit()
with tf.Session() as sess:
    nn = SingleLayerNeuralNet([2048], 2, 1024, sess=sess, name='InceptionV4Test')
    nn.train(train.X, train.y, epochs=10, sess=sess, verbose=True)
    _, acc = nn.validate(test.X, test.y, sess=sess)
    nn.save(model_path, sess=sess)

