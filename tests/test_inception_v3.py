import os
import cv2
import tensorflow as tf

from tfwrapper.nets.pretrained import InceptionV3

from utils import curr_path

cat_img = os.path.join(curr_path, 'data', 'cat.jpg')

class TestInceptionV3():

    @classmethod
    def setup_class(cls):
        cls.sess = tf.Session()
        cls.inception = InceptionV3()

    @classmethod
    def teardown_class(cls):
        cls.sess.close()

    def test_from_file(self):
        features = self.inception.extract_features_from_file(cat_img, sess=self.sess)

        assert (1, 1, 2048) == features.shape

    def test_from_data(self):
        img = cv2.imread(cat_img)
        features = self.inception.extract_features(img, sess=self.sess)

        assert (1, 1, 2048) == features.shape

    def test_from_op(self):
        img = cv2.imread(cat_img)
        features = self.inception.run_op(InceptionV3.DEFAULT_FEATURES_LAYER, InceptionV3.DEFAULT_INPUT_LAYER, img, sess=self.sess)

        assert (1, 1, 2048) == features.shape

    def test_bottleneck(self):
        img = cv2.imread(cat_img)
        features = self.inception.extract_bottleneck_features(img, sess=self.sess)

        assert (2048, ) == features.shape


    