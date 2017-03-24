import os
import cv2
import pytest
import numpy as np

from tfwrapper import ImageDataset
from tfwrapper import ImageTransformer

from utils import curr_path

cat_img = os.path.join(curr_path, 'data', 'cat.jpg')

def test_no_transformations():
	img = cv2.imread(cat_img)
	transformer = ImageTransformer()
	transformed_imgs, suffixes = transformer.transform(img)

	assert 1 == len(transformed_imgs)
	assert np.array_equal(img, transformed_imgs[0])
	assert 1 == len(suffixes)
	assert '' == suffixes[0]

def test_transform_resize():
	img = cv2.imread(cat_img)
	transformer = ImageTransformer(resize_to=(64, 64))
	transformed_imgs, suffixes = transformer.transform(img)

	assert 1 == len(transformed_imgs)
	assert (64, 64, 3) == transformed_imgs[0].shape
	assert 1 == len(suffixes)
	assert '' == suffixes[0]

def test_transform_bw():
	img = cv2.imread(cat_img)
	transformer = ImageTransformer(bw=True)
	transformed_imgs, suffixes = transformer.transform(img)

	assert 1 == len(transformed_imgs)
	assert np.array_equal(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), transformed_imgs[0])
	assert 1 == len(suffixes)
	assert '' == suffixes[0]

def test_transform_hflip():
	img = cv2.imread(cat_img)
	transformer = ImageTransformer(hflip=True)
	transformed_imgs, suffixes = transformer.transform(img)

	assert 2 == len(transformed_imgs)
	for transformed_img in transformed_imgs:
		assert np.array_equal(img, transformed_img) or \
				np.array_equal(np.fliplr(img), transformed_img)
	
	assert 2 == len(suffixes)
	assert '' in suffixes
	assert '_hflip' in suffixes

def test_transform_vflip():
	img = cv2.imread(cat_img)
	transformer = ImageTransformer(vflip=True)
	transformed_imgs, suffixes = transformer.transform(img)

	assert 2 == len(transformed_imgs)
	for transformed_img in transformed_imgs:
		assert np.array_equal(img, transformed_img) or \
				np.array_equal(np.flipud(img), transformed_img)
	
	assert 2 == len(suffixes)
	assert '' in suffixes
	assert '_vflip' in suffixes

def test_combinations():
	img = cv2.imread(cat_img)
	transformer = ImageTransformer(bw=True, hflip=True, vflip=True)
	transformed_imgs, suffixes = transformer.transform(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	assert 4 == len(transformed_imgs)
	for transformed_img in transformed_imgs:
		assert 2 == len(transformed_img.shape) or transformed_img.shape[2] == 1
		assert np.array_equal(img, transformed_img) or \
				np.array_equal(np.fliplr(img), transformed_img) or \
				np.array_equal(np.flipud(img), transformed_img) or \
				np.array_equal(np.fliplr(np.flipud(img)), transformed_img)

	assert 4 == len(suffixes)
	assert '' in suffixes
	assert '_vflip' in suffixes
	assert '_hflip' in suffixes
	assert '_hvflip' in suffixes

def test_dataset_with_transformation():
	X = []
	y = []
	names = []

	for i in range(1, 6):
		X.append(cv2.resize(cv2.imread(cat_img), (64*i, 64*i)))
		y.append(i)
		names.append('cat' + str(i) + '.jpg')

	dataset = ImageDataset(X=X, y=y, names=names)
	X, y, labels, names = dataset.getdata(transformer=ImageTransformer(resize_to=(64, 64), bw=True, hflip=True))

	assert 10 == len(X)
	assert 10 == len(y)
	assert 10 == len(names)

	for img in X:
		assert img.shape[0] == 64
		assert img.shape[1] == 64
		assert len(img.shape) == 2 or img.shape[2] == 1

	for i in range(0, 10, 2):
		assert y[i] == y[i + 1]


