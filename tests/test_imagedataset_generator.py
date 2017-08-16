import os

from tfwrapper.dataset import FeatureLoader
from tfwrapper.dataset import ImageDataset
from tfwrapper.dataset import ImageDatasetGenerator
from tfwrapper.models.frozen import FrozenInceptionV3

from utils import curr_path


def test_imagedataset_generator():
    dataset = ImageDataset.from_root_folder(os.path.join(curr_path, 'data', 'testset'))
    dataset.loader = FeatureLoader(FrozenInceptionV3())

    generator = ImageDatasetGenerator(dataset, 2)
    values = next(generator)

    assert 3 == len(values)
    assert 2 == len(values[2])
    assert values[2][0].endswith('cat.jpg')
    assert values[2][1].endswith('cat.jpg')


def test_imagedataset_generator_no_filenames():
    dataset = ImageDataset.from_root_folder(os.path.join(curr_path, 'data', 'testset'))
    dataset.loader = FeatureLoader(FrozenInceptionV3())

    generator = ImageDatasetGenerator(dataset, 2, include_filenames=False)

    assert 2 == len(next(generator))



