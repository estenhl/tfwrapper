import os
from tfwrapper.datasets import mnist
from tfwrapper.ensembles import TransferLearningEnsemble
from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.nets.pretrained import InceptionV3
from tfwrapper.nets.pretrained import InceptionV4

from utils import curr_path

dataset = mnist(size=50, verbose=True)
dataset = dataset.shuffle()
dataset = dataset.balance()
dataset = dataset.translate_labels()
dataset = dataset.onehot()

incv3 = InceptionV3()
incv4 = InceptionV4()
pretrained = [incv3, incv4]

model1 = SingleLayerNeuralNet([1536], 10, 1024, name='Model1')
model2 = SingleLayerNeuralNet([2048], 10, 1024, name='Model2')
models = [model1, model2]

feature_file1 = os.path.join(curr_path, 'data', 'inception_v3_flower_features.csv')
feature_file2 = os.path.join(curr_path, 'data', 'inception_v4_flower_features.csv')
feature_files = [feature_file1, feature_file2]

ensemble = TransferLearningEnsemble(pretrained, models, feature_files=feature_files)
ensemble.train(dataset, epochs=10)