import os

from tfwrapper.datasets import cats_and_dogs
from tfwrapper.validation import kfold_validation_imagesize

dataset = cats_and_dogs()
dataset = dataset.shuffle()
dataset = dataset[:10000]

validator = kfold_validation_imagesize(dataset, [(12, 12), (8, 8), (4, 4)], k=3)
print(str(validator))
validator.plot()