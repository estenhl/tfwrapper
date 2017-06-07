import os
import tensorflow as tf

from tfwrapper.datasets import flowers
from tfwrapper.models.frozen import FrozenInceptionV3
from tfwrapper.models.frozen import FrozenInceptionV4
from tfwrapper.models.frozen import FrozenResNet50
from tfwrapper.models.frozen import FrozenResNet152
from tfwrapper.models.frozen import FrozenVGG16
from tfwrapper.models.nets import SingleLayerNeuralNet
from tfwrapper.models import TransferLearningModel
from tfwrapper.ensembles import Stacker

from utils import curr_path

dataset = flowers(size=50)
#dataset = dataset.shuffle()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train = dataset #train, test = dataset.split(0.8)

datafolder = os.path.join(curr_path, 'data')
if not os.path.isdir(datafolder):
    os.mkdir(datafolder)
incv3_features_file = os.path.join(datafolder, 'flowers_inceptionv3.csv')
incv4_features_file = os.path.join(datafolder, 'flowers_inceptionv4.csv')
resnet50_features_file = os.path.join(datafolder, 'flowers_resnet50.csv')
resnet152_features_file = os.path.join(datafolder, 'flowers_resnet152.csv')
vgg16_features_file = os.path.join(datafolder, 'flowers_vgg16.csv')

models = []

inception_v3 = FrozenInceptionV3()
nn = SingleLayerNeuralNet([2048], 17, 1024, name='InceptionV3')
model = TransferLearningModel(inception_v3, nn, features_cache=incv3_features_file)
models.append(model)

inception_v4 = FrozenInceptionV4()
nn = SingleLayerNeuralNet([1536], 17, 1024, name='InceptionV4')
model = TransferLearningModel(inception_v4, nn, features_cache=incv4_features_file)
models.append(model)

resnet50 = FrozenResNet50()
nn = SingleLayerNeuralNet([2048], 17, 1024, name='ResNet50')
model = TransferLearningModel(resnet50, nn, features_cache=resnet50_features_file)
models.append(model)

resnet152 = FrozenResNet152()
nn = SingleLayerNeuralNet([2048], 17, 1024, name='ResNet152')
model = TransferLearningModel(resnet152, nn, features_cache=resnet152_features_file)
models.append(model)

vgg16 = FrozenVGG16()
nn = SingleLayerNeuralNet([4096], 17, 1024, name='VGG16')
model = TransferLearningModel(vgg16, nn, features_cache=vgg16_features_file)
models.append(model)

decision_model = SingleLayerNeuralNet([5], 17, 5, name='DecisionModel')
stacker = Stacker(models, decision_model)
stacker.train(train, epochs=10)