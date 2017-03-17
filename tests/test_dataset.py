import pytest
import numpy as np

from tfwrapper import Dataset

def test_normalize():
	X = np.asarray([5, 4, 3])
	y = np.asarray([1, 1, 1])
	dataset = Dataset(X=X, y=y)
	test_X, test_y, _ = dataset.getdata(normalize=True)
	
	assert 0 < test_X[0]
	assert 0 == test_X[1]
	assert 0 > test_X[2]
	assert test_X[0] == -test_X[2]
	assert np.array_equal(y, test_y)

def test_balance():
	X = np.zeros(100)
	y = np.concatenate([np.zeros(10), np.ones(90)])
	dataset = Dataset(X=X, y=y)
	test_X, test_y, _ = dataset.getdata(balance=True)

	assert 20 == len(test_X)
	assert 20 == len(test_y)
	assert 10 == np.sum(test_y)

def test_translate_labels():
	X = np.asarray([0, 1, 2])
	y = np.asarray(['Zero', 'One', 'Two'])
	dataset = Dataset(X=X, y=y)
	_, test_y, labels = dataset.getdata(translate_labels=True)

	assert 3 == len(labels)
	assert 'Zero' == labels[np.where(test_y==0)[0][0]]
	assert 'One' == labels[np.where(test_y==1)[0][0]]
	assert 'Two' == labels[np.where(test_y==2)[0][0]]

def test_shuffle():
	X = np.concatenate([np.zeros(100), np.ones(100)])
	y = np.concatenate([np.zeros(100), np.ones(100)])
	dataset = Dataset(X=X, y=y)
	test_X, test_y, _ = dataset.getdata(shuffle=True)

	assert not np.array_equal(X, test_X)
	assert np.array_equal(test_X, test_y)

def test_onehot():
	X = np.zeros(10)
	y = np.arange(10)
	dataset = Dataset(X=X, y=y)
	test_X, test_y, _ = dataset.getdata(onehot=True)

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
