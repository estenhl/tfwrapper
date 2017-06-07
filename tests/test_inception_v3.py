import os
import cv2
import tensorflow as tf

from tfwrapper.models.frozen import FrozenInceptionV3

from utils import curr_path

cat_img = os.path.join(curr_path, 'data', 'cat.jpg')

class TestInceptionV3():

    @classmethod
    def setup_class(cls):
        cls.sess = tf.Session()
        cls.inception = FrozenInceptionV3()

    @classmethod
    def teardown_class(cls):
        cls.sess.close()

    def test_from_data(self):
        img = cv2.imread(cat_img)
        features = self.inception.extract_features(img, sess=self.sess)

        assert (1, 1, 1, 2048) == features.shape

    def test_run_op(self):
        img = cv2.imread(cat_img)
        features = self.inception.run_op(img, dest=FrozenInceptionV3.bottleneck_tensor, src=FrozenInceptionV3.input_tensor, sess=self.sess)

        assert (1, 1, 1, 2048) == features.shape

    def test_bottleneck(self):
        img = cv2.imread(cat_img)
        features = self.inception.extract_bottleneck_features(img, sess=self.sess)

        assert (2048, ) == features.shape
