import os
import cv2
import tensorflow as tf

from tfwrapper.nets.pretrained import InceptionV4

from utils import curr_path

cat_img = os.path.join(curr_path, 'data', 'cat.jpg')

def test_from_file():
	inception = InceptionV4()
	features = inception.extract_features_from_file(cat_img)

	assert (1536, ) == features.shape

def test_from_data():
	inception = InceptionV4()
	img = cv2.imread(cat_img)
	features = inception.extract_features(img)

	assert (1536, ) == features.shape

def test_from_op():
    inception = InceptionV4()
    img = cv2.imread(cat_img)
    features = inception.run_op(InceptionV4.DEFAULT_FEATURES_LAYER, InceptionV4.DEFAULT_INPUT_LAYER, img)

    assert (1536, ) == features.shape

def test_bottleneck():
    inception = InceptionV4()
    img = cv2.imread(cat_img)
    features = inception.extract_bottleneck_features(img)

    assert (1536, ) == features.shape


	