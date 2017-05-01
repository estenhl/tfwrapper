import os
import cv2
import pytest
import numpy as np

from tfwrapper.utils.files import remove_dir
from tfwrapper.utils.images import copy_image
from tfwrapper.utils.images import copy_image_folder
from tfwrapper.utils.images import find_duplicates

from utils import curr_path

def create_image_dir(src=os.path.join(curr_path, 'tmp_src'), size=1, vary_colours=False):
	os.mkdir(src)

	img = np.zeros((4, 4, 3))
	for i in range(4):
		for j in range(4):
			img[i][j] = np.asarray(np.repeat(i * j, 3))
			if vary_colours:
				for k in range(3):
					img[i][j][k] = k

	imgs = []
	filenames = []
	for i in range(size):
		filename = os.path.join(src, str(i) + '.png')
		cv2.imwrite(os.path.join(src, str(i) + '.png'), img)
		filenames.append(filename)
		imgs.append(img)

	return src, imgs, filenames

def test_copy_image():
	src, imgs, filenames = create_image_dir()

	dest = os.path.join(curr_path, 'tmp_dest')
	os.mkdir(dest)
	src_file = filenames[0]

	copy_image(src_file, dest)

	dest_file = os.path.join(dest, os.path.basename(src_file))
	assert os.path.isfile(dest_file)

	copied_img = cv2.imread(dest_file).astype(float)
	remove_dir(src)
	remove_dir(dest)

	assert np.array_equal(imgs[0], copied_img)

def test_copy_image_resize():
	src, imgs, filenames = create_image_dir()

	dest = os.path.join(curr_path, 'tmp_dest')
	os.mkdir(dest)
	src_file = filenames[0]

	height, width, _ = imgs[0].shape
	copy_image(src_file, dest, size=(height * 2, width * 2))

	dest_file = os.path.join(dest, os.path.basename(src_file))
	copied_img = cv2.imread(dest_file).astype(float)

	remove_dir(src)
	remove_dir(dest)

	assert height * 2 == copied_img.shape[0]
	assert width * 2 == copied_img.shape[0]

def test_copy_image_bw():
	src, imgs, filenames = create_image_dir(vary_colours=True)

	dest = os.path.join(curr_path, 'tmp_dest')
	os.mkdir(dest)
	src_file = filenames[0]

	copy_image(src_file, dest, bw=True)

	dest_file = os.path.join(dest, os.path.basename(src_file))
	copied_img = cv2.imread(dest_file).astype(float)
	
	remove_dir(src)
	remove_dir(dest)

	img = imgs[0]
	assert img[0][0][0] != img[0][0][1]
	assert copied_img[0][0][0] == copied_img[0][0][1]

def test_copy_image_hflip():
	src, imgs, filenames = create_image_dir()

	dest = os.path.join(curr_path, 'tmp_dest')
	os.mkdir(dest)
	src_file = filenames[0]

	copy_image(src_file, dest, h_flip=True)

	prefix, _ = os.path.basename(src_file).split('.')
	dest_file = os.path.join(dest, prefix + '_hflip.png')
	copied_img = cv2.imread(dest_file).astype(float)
	
	remove_dir(src)
	remove_dir(dest)

	assert np.array_equal(imgs[0], np.fliplr(copied_img))

def test_copy_image_vflip():
	src, imgs, filenames = create_image_dir()

	dest = os.path.join(curr_path, 'tmp_dest')
	os.mkdir(dest)
	src_file = filenames[0]

	copy_image(src_file, dest, v_flip=True)

	prefix, _ = os.path.basename(src_file).split('.')
	dest_file = os.path.join(dest, prefix + '_vflip.png')
	copied_img = cv2.imread(dest_file).astype(float)
	
	remove_dir(src)
	remove_dir(dest)

	assert np.array_equal(imgs[0], np.flipud(copied_img))

def test_copy_images():
	src, imgs, filenames = create_image_dir(size=10)

	dest = os.path.join(curr_path, 'tmp_dest')
	os.mkdir(dest)

	copy_image_folder(src, dest)

	assert len(imgs) == len(os.listdir(dest))

	remove_dir(src)
	remove_dir(dest)

def test_find_duplicates():
	src, imgs, filenames = create_image_dir(size=2)

	for i in range(2):
		img = np.zeros((10, 10, 3))
		cv2.imwrite(os.path.join(src, 'duplicate' + str(i) + '.png'), img)

	for i in range(1, 10):
		img = np.ones((10, 10, 3)) * i
		cv2.imwrite(os.path.join(src, 'non_duplicate' + str(i) + '.png'), img)

	duplicates = find_duplicates(src)

	assert 13 == len(os.listdir(src))

	remove_dir(src)

	assert 2 == len(duplicates)
