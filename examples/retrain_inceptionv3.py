import os
import tensorflow as tf

from tfwrapper import Dataset
from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.nets.pretrained import InceptionV3
from tfwrapper.datasets import checkboxes

from utils import curr_path

features_path = os.path.join(curr_path, 'data', 'checkbox_features.csv')
model_path = os.path.join(curr_path, 'data', 'inception')
if not os.path.isdir(os.path.join(curr_path, 'data')):
	os.mkdir(os.path.join(curr_path, 'data'))

_, data_folder = checkboxes()
inception = InceptionV3()
features = inception.extract_features_from_datastructure(data_folder, feature_file=features_path)

dataset = Dataset(features=features)
dataset = dataset.normalize()
dataset = dataset.balance()
dataset = dataset.translate_labels()
dataset = dataset.shuffle()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

nn = SingleLayerNeuralNet([train.X.shape[1]], 2, 1024, name='InceptionV3Test')
nn.train(train.X, train.y, epochs=10, verbose=True)
nn.save(model_path)
_, acc = nn.validate(test.X, test.y)
print('Inception Test accuracy: %d %%' % (acc*100))