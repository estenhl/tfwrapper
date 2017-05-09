import os
import cv2
import tensorflow as tf

from tfwrapper.nets.pretrained import InceptionV4

from utils import curr_path

cat_img = os.path.join(curr_path, 'data', 'cat.jpg')
inception = InceptionV4()
sess = tf.Session()

def test_from_file():
	features = inception.extract_features_from_file(cat_img, sess=sess)

	assert (1536, ) == features.shape

def test_from_data():
	img = cv2.imread(cat_img)
	features = inception.extract_features(img, sess=sess)

	assert (1536, ) == features.shape

def test_from_op():
    img = cv2.imread(cat_img)
    features = inception.run_op(InceptionV4.DEFAULT_FEATURES_LAYER, InceptionV4.DEFAULT_INPUT_LAYER, img, sess=sess)

    assert (1536, ) == features.shape

def test_bottleneck():
    img = cv2.imread(cat_img)
    features = inception.extract_bottleneck_features(img, sess=sess)

    assert (1536, ) == features.shape


	