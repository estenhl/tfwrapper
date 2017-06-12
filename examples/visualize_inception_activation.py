import os
import numpy as np

from tfwrapper import ImageDataset
from tfwrapper.models.frozen import FrozenInceptionV3
from tfwrapper.visualization import visualize_activation
from tfwrapper.datasets import cats_and_dogs

dataset = cats_and_dogs()

img = dataset[0].X[0]

inception = FrozenInceptionV3()
features = inception.extract_features(img, dest='mixed_10/join:0')
features = np.reshape(features, (8, 8, 2048))

visualize_activation(features, img)