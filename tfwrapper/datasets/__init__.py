import os
import cv2
import gzip
import tarfile
import zipfile
import urllib.request
import numpy as np
from shutil import copyfile
from struct import unpack

from tfwrapper import config
from tfwrapper import Dataset
from tfwrapper import ImageDataset
from tfwrapper.utils.files import download_file

from .utils import setup_structure
from .utils import recursive_delete
from .utils import curr_path
from .mnist import parse_mnist
from .cifar import parse_cifar10
from .cifar import parse_cifar100
from .catsdogs import download_cats_and_dogs

curr_path = config.DATASETS

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

			if filename.lower().endswith('.jpg'):
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

def cats_and_dogs():
	data_path = download_cats_and_dogs()
	dataset = ImageDataset(root_folder=data_path)

	return dataset

def mnist(size=None, imagesize=[28, 28]):
	X, y = parse_mnist(size=size, imagesize=imagesize)
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

def cifar10(size=50000):
	X, y = parse_cifar10(size=size)
	dataset = Dataset(X=X, y=y)
	return dataset

def cifar100(size=50000):
	X, y = parse_cifar100(size=size)
	dataset = Dataset(X=X, y=y)
	return dataset
