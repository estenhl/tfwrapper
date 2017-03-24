import os
import cv2
import pytest
import numpy as np

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
