import os

from tfwrapper.dataset import ImageDataset
from tfwrapper.dataset import ImageDatasetGenerator

from utils import curr_path


def test_imagedataset_generator():
    dataset = ImageDataset.from_root_folder(os.path.join(curr_path, 'data', 'testset'))

    generator = ImageDatasetGenerator(dataset, 2)

    assert 3 == len(next(generator))


def test_imagedataset_generator_no_filenames():
    dataset = ImageDataset.from_root_folder(os.path.join(curr_path, 'data', 'testset'))

    generator = ImageDatasetGenerator(dataset, 2, include_filenames=False)

    assert 2 == len(next(generator))


