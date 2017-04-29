import os
import cv2
import gzip
import tarfile
import zipfile
import urllib.request
import numpy as np

from numpy import zeros, uint8, float32
from shutil import copyfile
from struct import unpack

from tfwrapper import config
from tfwrapper import Dataset
from tfwrapper import ImageDataset
from tfwrapper.utils.files import download_file


curr_path = config.DATASETS

def setup_structure(name, create_data_folder=True):
	root_path = os.path.join(curr_path, name)
	if not os.path.isdir(root_path):
		os.mkdir(root_path)

	data_path = os.path.join(root_path, 'data')
	if create_data_folder and not os.path.isdir(data_path):
		os.mkdir(data_path)

	labels_file = os.path.join(root_path, 'labels.txt')

	return root_path, data_path, labels_file

def recursive_delete(path, skip=[]):
	ret_val = True
	if os.path.isdir(path):
		for filename in os.listdir(path):
			ret_val = recursive_delete(os.path.join(path, filename), skip=skip) and ret_val


		if ret_val:
			os.rmdir(path)

		return ret_val
	elif os.path.isfile(path) and path not in skip:
		os.remove(path)

		return True

	return False

def download_cats_and_dogs(verbose=False):
	root_path, data_path, labels_file = setup_structure('cats_and_dogs')

	num_refined_files = len(os.listdir(data_path))
	num_raw_files = 0
	if os.path.isdir(os.path.join(root_path, 'train')):
		num_raw_files = len(os.listdir(os.path.join(root_path, 'train')))
	
	if num_refined_files < 25000:
		local_zip = os.path.join(root_path, 'train.zip')
		if not os.path.isfile(local_zip):
			print('Downloading dataset cats_and_dogs')
			raise NotImplementedError('Unable to download zip')

		if num_raw_files < 25000:
			with zipfile.ZipFile(local_zip, 'r') as f:
				print('Extracting dataset cats_and_dogs')
				f.extractall(root_path)

		print('Refining dataset cats_and_dogs')
		with open(labels_file, 'w') as f:
			for filename in os.listdir(os.path.join(root_path, 'train')):
				src_file = os.path.join(root_path, 'train', filename)

				if not filename.endswith('.jpg'):
					os.remove(src_file)
					continue

				pref, midf, _ = filename.split('.')
				dest = pref + '_' + midf + '.jpg'
				dest_file = os.path.join(data_path, dest)

				if filename.startswith('cat'):
					f.write('cat,' + dest + '\n')
				elif filename.startswith('dog'):
					f.write('dog,' + dest + '\n')
				else:
					os.remove(src_file)
					continue

				img = cv2.imread(src_file)
				img = cv2.resize(img, (196, 196))
				cv2.imwrite(dest_file, img)
				os.remove(src_file)

		os.rmdir(os.path.join(root_path, 'train'))
		if os.path.isfile(os.path.join(root_path, 'train.zip')):
			os.remove(os.path.join(root_path, 'train.zip'))
	
	return data_path, labels_file

def parse_mnist_data(data_file, labels_file, size=None, verbose=False):
	if verbose:
		print('Unpacking mnist data')

	images = gzip.open(data_file, 'rb')
	labels = gzip.open(labels_file, 'rb')

	images.read(4)
	number_of_images = images.read(4)
	number_of_images = unpack('>I', number_of_images)[0]
	rows = images.read(4)
	rows = unpack('>I', rows)[0]
	cols = images.read(4)
	cols = unpack('>I', cols)[0]

	labels.read(4)
	N = labels.read(4)
	N = unpack('>I', N)[0]
	if size is not None:
		N = size

	X = zeros((N, rows, cols), dtype=float32)
	y = zeros((N, 1), dtype=uint8)
	for i in range(N):
		if i % 1000 == 0 and verbose:
			print("Read %i of %i images" % (i, N))
		for row in range(rows):
			for col in range(cols):
				tmp_pixel = images.read(1)
				tmp_pixel = unpack('>B', tmp_pixel)[0]
				X[i][row][col] = tmp_pixel
		tmp_label = labels.read(1)
		y[i] = unpack('>B', tmp_label)[0]
	if verbose:
		print('Read %i images' % N)

	return X, y

def download_mnist(size=None, verbose=False):
	data_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
	labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'

	root_path, data_path, labels_file = setup_structure('mnist')
	local_data_zip = os.path.join(root_path, 'data.zip')
	local_labels_zip = os.path.join(root_path, 'labels.zip')

	num_files = len(os.listdir(data_path))
	if True: #num_files < zzz
		if not os.path.isfile(local_data_zip):
			download_file(data_url, local_data_zip, verbose=verbose)
		if not os.path.isfile(local_labels_zip):
			download_file(labels_url, local_labels_zip, verbose=verbose)

	X, y = parse_mnist_data(local_data_zip, local_labels_zip, size=size, verbose=verbose)
	
	return X, y

def download_ptb(verbose=False):
	url = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
	root_path, _, labels_file = setup_structure('penn_tree_bank', create_data_folder=False)
	local_tar = os.path.join(root_path, 'simple-examples.tgz')

	examples_path = os.path.join(root_path, 'simple-examples')
	data_path = os.path.join(examples_path, 'data')
	data_file = os.path.join(data_path, 'ptb.train.txt')
	if not (os.path.isdir(examples_path) and os.path.isdir(data_path) and os.path.isfile(data_file)):
		if not os.path.isfile(local_tar):
			download_file(url, local_tar, verbose=verbose)

		with tarfile.open(local_tar, 'r') as f:
			if verbose:
				print('Extracting penn_tree_bank data')
			for item in f:
				f.extract(item, root_path)

		for folder in os.listdir(root_path):
			recursive_delete(os.path.join(root_path, folder), skip=[data_file])

	return data_file

def download_flowers(verbose=False):
	url = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz'

	root_path, data_folder, labels_file = setup_structure('flowers')

	if not len(os.listdir(data_folder)) > 1000:
		tgz_file = os.path.join(root_path, '17flowers.tgz')
		if not os.path.isfile(tgz_file):
			download_file(url, tgz_file, verbose=verbose)

		with tarfile.open(tgz_file, 'r') as f:
			if verbose:
				print('Extracting flowers data')
			for item in f:
				f.extract(item, root_path)

		tmp_folder = os.path.join(root_path, 'jpg')
		for filename in os.listdir(tmp_folder):
			src = os.path.join(tmp_folder, filename)

			if filename.endswith('.jpg'):
				dest = os.path.join(data_folder, filename)
				copyfile(src, dest)

			os.remove(src)
		os.rmdir(tmp_folder)
		os.remove(tgz_file)

	if not os.path.isfile(labels_file):
		labels = ['flower' + str(x) for x in range(17)]
		with open(labels_file, 'w') as f:
			for i in range(0, 17):
				for j in range(1, 80):
					index = str((i * 80) + j)
					while len(index) < 4:
						index = '0' + index
					f.write('%s,image_%s.jpg\r\n' % (labels[i], index))

	return data_folder, labels_file


def checkboxes(verbose=False):
	raise NotImplementedError('Deprecated dataset')

def cats_and_dogs(verbose=False):
	data_path, labels_file = download_cats_and_dogs(verbose=verbose)
	dataset = ImageDataset(root_folder=data_path, labels_file=labels_file, verbose=verbose)
	return dataset

def mnist(size=None, verbose=False):
	X, y = download_mnist(size=size, verbose=verbose)
	dataset = Dataset(X=X, y=np.asarray(y).flatten())
	return dataset

def flowers(size=1360, verbose=False):
	data_path, labels_file = download_flowers(verbose=verbose)

	if size < 1360:
		tmp_labels_file = os.path.join(os.path.dirname(labels_file), 'tmp.txt')
		with open(labels_file, 'r') as f:
			lines = f.readlines()

		num_classes = 17
		num_flowers_per_class = int(size / num_classes)
		total_flowers_per_class = 80
		with open(tmp_labels_file, 'w') as f:
			for i in range(num_classes):
				for j in range(num_flowers_per_class):
					index = (i * total_flowers_per_class) + j
					f.write(lines[index])
		dataset = ImageDataset(root_folder=data_path, labels_file=tmp_labels_file)
		os.remove(tmp_labels_file)
	else:
		dataset = ImageDataset(root_folder=data_path, labels_file=labels_file)
	
	return dataset
