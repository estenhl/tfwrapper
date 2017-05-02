import os

from tfwrapper import ImageDataset
from tfwrapper.validation import kfold_validation_imagesize

curr_path = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
data_path = os.path.join(curr_path, '..', 'data', 'datasets', 'catsdogs', 'images')

dataset = ImageDataset(root_folder=data_path)
dataset = dataset.shuffle()
dataset = dataset[:10000]

validator = kfold_validation_imagesize(dataset, [(12, 12), (8, 8), (4, 4)], k=3)
print(str(validator))
validator.plot()