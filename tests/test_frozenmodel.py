import os
import pytest
import tensorflow as tf

from tfwrapper import twimage
from tfwrapper.models import FrozenModel
from tfwrapper.models.frozen import FrozenVGG16
from tfwrapper.models.frozen import FrozenInceptionV3
from tfwrapper.models.frozen import FrozenInceptionV4
from tfwrapper.models.frozen import FrozenResNet50
from tfwrapper.models.frozen import FrozenResNet152
from tfwrapper.models.frozen.utils import VGG16_PB_PATH
from tfwrapper.models.frozen.utils import INCEPTIONV3_PB_PATH
from tfwrapper.models.frozen.utils import INCEPTIONV4_PB_PATH
from tfwrapper.models.frozen.utils import RESNET50_PB_PATH
from tfwrapper.models.frozen.utils import RESNET152_PB_PATH
from tfwrapper.models.frozen import FrozenVGG16
from tfwrapper.utils.exceptions import InvalidArgumentException

from utils import curr_path
from utils import is_better_than_tensorflow_errormsg

cat_img = twimage.imread(os.path.join(curr_path, 'data', 'cat.jpg'))

class TestInceptionV3():

    @classmethod
    def setup_class(cls):
        cls.sess = tf.Session()

    @classmethod
    def teardown_class(cls):
        cls.sess.close()

    def test_invalid_path(self):
        exception = False

        try:
            model = FrozenModel(path='not/a/valid/path', sess=self.sess)
        except InvalidArgumentException as e:
            exception = True
            errormsg = str(e)

        assert exception, 'Creating FrozenModel with invalid path does not raise an exception'
        assert is_better_than_tensorflow_errormsg(errormsg)

    def test_vgg16(self):
        if os.path.isfile(VGG16_PB_PATH):
            model = FrozenVGG16(sess=self.sess)
            features = model.extract_bottleneck_features(cat_img, sess=self.sess)

            assert (4096, ) == features.shape, 'FrozenVGG16 yields unexpected shape %s' % str(features.shape)

    def test_inception_v3(self):
        if os.path.isfile(INCEPTIONV3_PB_PATH):
            model = FrozenInceptionV3(sess=self.sess)
            features = model.extract_bottleneck_features(cat_img, sess=self.sess)

            assert (2048, ) == features.shape, 'FrozenInceptionV3 yields unexpected shape %s' % str(features.shape)

    def test_inception_v4(self):
        if os.path.isfile(INCEPTIONV4_PB_PATH):
            model = FrozenInceptionV4(sess=self.sess)
            features = model.extract_bottleneck_features(cat_img, sess=self.sess)

            assert (1536, ) == features.shape, 'FrozenInceptionV4 yields unexpected shape %s' % str(features.shape)

    def test_resnet50(self):
        if os.path.isfile(RESNET50_PB_PATH):
            model = FrozenResNet50(sess=self.sess)
            features = model.extract_bottleneck_features(cat_img, sess=self.sess)

            assert (2048, ) == features.shape, 'FrozenResNet50 yields unexpected shape %s' % str(features.shape)

    def test_resnet152(self):
        if os.path.isfile(RESNET152_PB_PATH):
            model = FrozenResNet152(sess=self.sess)
            features = model.extract_bottleneck_features(cat_img, sess=self.sess)

            assert (2048, ) == features.shape, 'FrozenResNet152 yields unexpected shape %s' % str(features.shape)

    def test_name(self):
        if os.path.isfile(INCEPTIONV3_PB_PATH):
            name = 'test'
            model = FrozenModel(path=INCEPTIONV3_PB_PATH, name=name, sess=self.sess)

            assert name == model.name, 'FrozenModel does not get given name'

    def test_from_type(self):
        if os.path.isfile(INCEPTIONV3_PB_PATH):
            name = 'test'
            model = FrozenModel.from_type('FrozenInceptionV3', name=name, sess=self.sess)

            assert FrozenInceptionV3 == type(model), 'FrozenModel.from_type returns the wrong subclass (Expected FrozenInceptionV3, got %s)' % str(type(model))
            assert name == model.name

        exception = False
        try:
            model = FrozenModel.from_type('Invalid type', sess=self.sess)
        except InvalidArgumentException:
            exception = True

        assert exception, 'Able to call FrozenModel.from_type with invalid type'