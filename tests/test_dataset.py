import os
import cv2
import pytest
import numpy as np

from tfwrapper import Dataset
from tfwrapper import ImageDataset
from tfwrapper.utils.data import write_features

from utils import curr_path
from utils import remove_dir
from utils import generate_features

def test_create_from_data():
	X = np.asarray([1, 2, 3])
	y = np.asarray([2, 4, 6])
	dataset = Dataset(X=X, y=y)
	test_X, test_y, _, _, [] = dataset.getdata()

	assert np.array_equal(X, test_X)
	assert np.array_equal(y, test_y)

def test_create_from_features():
	X, y, features = generate_features()
	dataset = Dataset(features=features)
	test_X, test_y, _, _, _ = dataset.getdata()

	assert np.array_equal(X, test_X)
	assert np.array_equal(y, test_y)

def test_create_from_feature_file():
	X, y, features = generate_features()
	tmp_file = os.path.join(curr_path, 'tmp.csv')
	write_features(tmp_file, features)
	dataset = Dataset(features_file=tmp_file)
	os.remove(tmp_file)
	test_X, test_y, _, _, _ = dataset.getdata()

	assert np.array_equal(X, test_X)
	assert np.array_equal(y, test_y)

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
	remove_dir(root_folder)
	X, y, _, _, _, _ = dataset.getdata()


	assert size == len(X)
	assert size == len(y)

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
	remove_dir(parent)
	os.remove(labels_file)
	X, y, _, _, _, _ = dataset.getdata()


	assert size / 2 == len(X)
	assert size / 2 == len(y)

def test_normalize():
	X = np.asarray([5, 4, 3])
	y = np.asarray([1, 1, 1])
	dataset = Dataset(X=X, y=y)
	test_X, test_y, _, _, _ = dataset.getdata(normalize=True)
	
	assert 0 < test_X[0]
	assert 0 == test_X[1]
	assert 0 > test_X[2]
	assert test_X[0] == -test_X[2]
	assert np.array_equal(y, test_y)

def test_balance():
	X = np.zeros(100)
	y = np.concatenate([np.zeros(10), np.ones(90)])
	dataset = Dataset(X=X, y=y)
	test_X, test_y, _, _, _ = dataset.getdata(balance=True)

	assert 20 == len(test_X)
	assert 20 == len(test_y)
	assert 10 == np.sum(test_y)

def test_translate_labels():
	X = np.asarray([0, 1, 2])
	y = np.asarray(['Zero', 'One', 'Two'])
	dataset = Dataset(X=X, y=y)
	_, test_y, _, _, labels = dataset.getdata(translate_labels=True)

	assert 3 == len(labels)
	assert 'Zero' == labels[np.where(test_y==0)[0][0]]
	assert 'One' == labels[np.where(test_y==1)[0][0]]
	assert 'Two' == labels[np.where(test_y==2)[0][0]]

def test_shuffle():
	X = np.concatenate([np.zeros(100), np.ones(100)])
	y = np.concatenate([np.zeros(100), np.ones(100)])
	dataset = Dataset(X=X, y=y)
	test_X, test_y, _, _, _ = dataset.getdata(shuffle=True)

	assert not np.array_equal(X, test_X)
	assert np.array_equal(test_X, test_y)

def test_onehot():
	X = np.zeros(10)
	y = np.arange(10)
	dataset = Dataset(X=X, y=y)
	test_X, test_y, _, _, _ = dataset.getdata(onehot=True)

	assert (10, 10) == test_y.shape
	for i in range(10):
		arr = np.zeros(10)
		arr[i] = 1
		assert np.array_equal(arr, test_y[i])

def test_split():
	X = np.concatenate([np.zeros(80), np.ones(20)])
	y = np.concatenate([np.zeros(80), np.ones(20)])
	dataset = Dataset(X=X, y=y)
	test_X, test_y, val_X, val_y, labels = dataset.getdata(split=True)

	assert np.array_equal(test_X, np.zeros(80))
	assert np.array_equal(test_y, np.zeros(80))
	assert np.array_equal(val_X, np.ones(20))
	assert np.array_equal(val_y, np.ones(20))
