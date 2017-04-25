import os
import cv2
import tensorflow as tf

from tfwrapper.nets.pretrained import InceptionV3

from utils import curr_path

cat_img = os.path.join(curr_path, 'data', 'cat.jpg')

def test_pool3_from_file():
	inception = InceptionV3()
	features = inception.extract_features_from_file(cat_img)

	assert (2048, ) == features.shape

def test_pool3_from_data():
	inception = InceptionV3()
	img = cv2.imread(cat_img)
	features = inception.extract_features_from_img(img)

	assert (2048, ) == features.shape

	