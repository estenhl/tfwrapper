import os
import pytest
import numpy as np

from tfwrapper import twimage
from tfwrapper.models.frozen import FrozenInceptionV3
from tfwrapper import FeatureLoader

from utils import curr_path


cat_img = os.path.join(curr_path, 'data', 'cat.jpg')
dog_img = os.path.join(curr_path, 'data', 'dog.jpg')


def test_featureloader_with_cache():
    feature_file = 'tmp.csv'
    try:
        loader = FeatureLoader(FrozenInceptionV3(), cache=feature_file)

        cat_features = loader.load(cat_img, name='test', label='cat')
        dog_features = loader.load(dog_img, name='test', label='dog')

        assert np.array_equal(cat_features, dog_features), 'Features are not cached on name when cache is set'
    finally:
        os.remove(feature_file)


def test_featureloader_no_cache():
    loader = FeatureLoader(FrozenInceptionV3())

    cat_features = loader.load(cat_img, name='test')
    dog_features = loader.load(dog_img, name='test')

    assert not np.array_equal(cat_features, dog_features), 'Features are cached even when cache=None'