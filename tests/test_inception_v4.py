import os
import cv2
import tensorflow as tf

from tfwrapper.models.frozen import FrozenInceptionV4

from utils import curr_path

cat_img = os.path.join(curr_path, 'data', 'cat.jpg')

class TestInceptionV4():

    @classmethod
    def setup_class(cls):
        cls.sess = tf.Session()
        cls.inception = FrozenInceptionV4(sess=cls.sess)

    @classmethod
    def teardown_class(cls):
        cls.sess.close()

    def test_from_data(self):
        img = cv2.imread(cat_img)
        features = self.inception.extract_features(img, sess=self.sess)

        assert (1, 1536) == features.shape

    def test_from_op(self):
        img = cv2.imread(cat_img)
        features = self.inception.run_op(img, src=FrozenInceptionV4.input_tensor, dest=FrozenInceptionV4.bottleneck_tensor, sess=self.sess)

        assert (1, 1536) == features.shape

    def test_bottleneck(self):
        img = cv2.imread(cat_img)
        features = self.inception.extract_bottleneck_features(img, sess=self.sess)

        assert (1536, ) == features.shape


    