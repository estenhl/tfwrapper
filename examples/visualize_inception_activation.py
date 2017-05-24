import os
import numpy as np

from tfwrapper import ImageDataset
from tfwrapper.nets.pretrained import InceptionV3
from tfwrapper.visualization import visualize_activation

from utils import curr_path

data_path = os.path.join(curr_path, '..', 'data', 'datasets', 'catsdogs', 'images')
dataset = ImageDataset(root_folder=data_path)

img = dataset[0].X[0]

inception = InceptionV3()
features = inception.extract_features(img, layer='mixed_10/join:0')
features = np.reshape(features, (8, 8, 2048))
print('Features shape: %s' % str(features.shape))

visualize_activation(features, img)