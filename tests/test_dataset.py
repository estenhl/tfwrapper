import os
import cv2
import pytest
import numpy as np

from tfwrapper import Dataset
from tfwrapper import ImageDataset
from tfwrapper.utils.data import write_features

from .utils import curr_path
from .utils import remove_dir
from .utils import generate_features

def test_create_from_data():
	X = np.asarray([1, 2, 3])
	y = np.asarray([2, 4, 6])
	dataset = Dataset(X=X, y=y)

	assert np.array_equal(X, dataset.X)
	assert np.array_equal(y, dataset.y)

def test_create_from_features():
	X, y, features = generate_features()
	dataset = Dataset(features=features)

	assert np.array_equal(X, dataset.X)
	assert np.array_equal(y, dataset.y)

def test_create_from_feature_file():
	X, y, features = generate_features()
	tmp_file = os.path.join(curr_path, 'tmp.csv')
	write_features(tmp_file, features)
	dataset = Dataset(features_file=tmp_file)
	os.remove(tmp_file)

	assert np.array_equal(X, dataset.X)
	assert np.array_equal(y, dataset.y)

def create_tmp_dir(root=os.path.join(curr_path, 'tmp'), size=10):
	os.mkdir(root)
	for label in ['x', 'y']:
		os.mkdir(os.path.join(root, label))
		for i in range(int(size/2)):
			img = np.zeros((10, 10, 3))
			path = os.path.join(root, label, str(i) + '.jpg')
			cv2.imwrite(path, img)

	return root

def test_create_from_datastructure():
	size = 10
	root_folder = create_tmp_dir(size=size)
	dataset = ImageDataset(root_folder=root_folder)

	assert size == len(dataset)
	assert size == len(dataset.X)
	assert size == len(dataset.y)

	remove_dir(root_folder)

def create_tmp_labels_file(root_folder, name):
	with open(name, 'w') as f:
		for filename in os.listdir(root_folder):
			f.write('label,' + filename + '\n')

	return name

def test_create_from_labels_file():
	size = 10
	parent = create_tmp_dir(size=size)
	root_folder = os.path.join(parent, 'x')
	labels_file = os.path.join(curr_path, 'tmp.csv')
	labels_file = create_tmp_labels_file(root_folder, labels_file)

	dataset = ImageDataset(root_folder=root_folder, labels_file=labels_file)

	assert size / 2 == len(dataset.X)
	assert size / 2 == len(dataset.y)

	remove_dir(parent)
	os.remove(labels_file)

def test_normalize():
	X = np.asarray([5, 4, 3])
	y = np.asarray([1, 1, 1])
	dataset = Dataset(X=X, y=y)
	dataset = dataset.normalize()
	
	assert 0 < dataset.X[0]
	assert 0 == dataset.X[1]
	assert 0 > dataset.X[2]
	assert dataset.X[0] == -dataset.X[2]
	assert np.array_equal(y, dataset.y)

def test_balance():
	X = np.zeros(100)
	y = np.concatenate([np.zeros(10), np.ones(90)])
	dataset = Dataset(X=X, y=y)
	dataset = dataset.balance()

	assert 20 == len(dataset.X)
	assert 20 == len(dataset.y)
	assert 10 == np.sum(dataset.y)

def test_translate_labels():
	X = np.asarray([0, 1, 2])
	y = np.asarray(['Zero', 'One', 'Two'])
	dataset = Dataset(X=X, y=y)
	dataset = dataset.translate_labels()
	labels = dataset.labels

	assert 3 == len(dataset.labels)
	assert 'Zero' == labels[dataset.y[np.where(dataset.X==0)[0][0]]]
	assert 'One' == labels[dataset.y[np.where(dataset.X==1)[0][0]]]
	assert 'Two' == labels[dataset.y[np.where(dataset.X==2)[0][0]]]

def test_shuffle():
	X = np.concatenate([np.zeros(100), np.ones(100)])
	y = np.concatenate([np.zeros(100), np.ones(100)])
	dataset = Dataset(X=X, y=y)
	dataset = dataset.shuffle()

	assert not np.array_equal(X, dataset.X)
	assert np.array_equal(dataset.X, dataset.y)

def test_onehot():
	X = np.zeros(10)
	y = np.arange(10)
	dataset = Dataset(X=X, y=y)
	dataset = dataset.onehot()

	assert (10, 10) == dataset.y.shape
	for i in range(10):
		arr = np.zeros(10)
		arr[i] = 1
		assert np.array_equal(arr, dataset.y[i])

def test_split():
	X = np.concatenate([np.zeros(80), np.ones(20)])
	y = np.concatenate([np.zeros(80), np.ones(20)])
	dataset = Dataset(X=X, y=y)
	train_dataset, test_dataset = dataset.split(ratio=0.8)

	assert np.array_equal(train_dataset.X, np.zeros(80))
	assert np.array_equal(train_dataset.y, np.zeros(80))
	assert np.array_equal(test_dataset.X, np.ones(20))
	assert np.array_equal(test_dataset.y, np.ones(20))

def test_length():
	X = np.zeros(100)
	y = np.zeros(100)
	dataset = Dataset(X=X, y=y)

	assert len(X) == len(dataset)

def test_drop_classes():
	X = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
	y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
	dataset = Dataset(X=X, y=y)
	dataset = dataset.drop_classes(drop=[2, 3])

	assert 6 == len(dataset.X)
	assert 6 == len(dataset.y)

def test_keep_classes():
	X = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
	y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
	dataset = Dataset(X=X, y=y)
	dataset = dataset.drop_classes(keep=[1, 2, 3])

	assert 9 == len(dataset.X)
	assert 9 == len(dataset.y)

def test_add_datasets():
	X1 = np.arange(10)
	y1 = np.arange(10)
	X2 = np.arange(10) + 10
	y2 = np.arange(10) + 10

	dataset1 = Dataset(X=X1, y=y1)
	dataset2 = Dataset(X=X2, y=y2)
	dataset = dataset1 + dataset2

	assert len(X1) + len(X2) == len(dataset)
	for i in range(20):
		assert i in dataset.X
		assert dataset.X[i] == dataset.y[i]

def test_slice_datasets():
	X = np.arange(10)
	y = np.arange(10)
	dataset = Dataset(X=X, y=y)
	dataset = dataset[:5]

	assert 5 == len(dataset)
	for i in range(5):
		assert i in dataset.X
		assert dataset.X[i] == dataset.y[i]

def test_index_datasets():
	X = np.arange(10)
	y = np.arange(10)
	dataset = Dataset(X=X, y=y)
	dataset = dataset[5]

	assert 1 == len(dataset)
	assert np.array_equal(np.asarray([5]), dataset.X)
	assert np.array_equal(np.asarray([5]), dataset.y)
