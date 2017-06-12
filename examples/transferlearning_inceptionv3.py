import os
import tensorflow as tf

from tfwrapper import ImagePreprocessor
from tfwrapper.models import TransferLearningModel
from tfwrapper.models.nets import SingleLayerNeuralNet
from tfwrapper.models.frozen import FrozenInceptionV3
from tfwrapper.datasets import cats_and_dogs

from utils import curr_path

dataset = cats_and_dogs(size=500)
dataset = dataset.balance(max=150)
dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

datafolder = os.path.join(curr_path, 'data')
if not os.path.isdir(datafolder):
    os.mkdir(datafolder)
features_file = os.path.join(datafolder, 'catsdogs_inceptionv3.csv')

inception = FrozenInceptionV3()
nn = SingleLayerNeuralNet([2048], 2, 1024, name='InceptionV3Test')
model = TransferLearningModel(inception, nn, features_cache=features_file)

preprocessor = ImagePreprocessor()
preprocessor.flip_lr = True
model.train(train, epochs=10, preprocessor=preprocessor)

print('Predicting')
preds = model.predict(test)
print('Validating')
_, acc = model.validate(test)
print('Acc: %d%%' % (acc * 100))