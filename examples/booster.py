import os
import tensorflow as tf

from tfwrapper import ImageDataset
from tfwrapper import FeatureLoader
from tfwrapper import ImagePreprocessor
from tfwrapper.models.nets import SingleLayerNeuralNet
from tfwrapper.models.frozen import FrozenInceptionV3
from tfwrapper.ensembles import Booster
from tfwrapper.datasets import cats_and_dogs

from utils import curr_path

dataset = cats_and_dogs()
dataset = dataset.balance(max=500)
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

datafolder = os.path.join(curr_path, 'data')
if not os.path.isdir(datafolder):
    os.mkdir(datafolder)
features_file = os.path.join(datafolder, 'catsdogs_inceptionv3.csv')

inception = FrozenInceptionV3()

train_prep = ImagePreprocessor()
train_prep.resize_to = (299, 299)
train_prep.flip_lr = True
train_loader = FeatureLoader(inception, cache=features_file, preprocessor=train_prep)
train.loader = train_loader

test_prep = ImagePreprocessor()
test_prep.resize_to = (299, 299)
test_loader = FeatureLoader(inception, cache=features_file, preprocessor=test_prep)
test.loader = test_loader

models = []
for i in range(3):
    models.append(SingleLayerNeuralNet([2048], 2, 1024, name='InceptionV3Test'))


ensemble = Booster(models)
ensemble.train(train, epochs=10)
_, acc = ensemble.validate(test.X, test.y)
print('Acc: %d%%' % (acc * 100))
