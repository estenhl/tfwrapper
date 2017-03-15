import os
import cv2
import zipfile
from tfwrapper import Dataset

curr_path = os.path.dirname(os.path.realpath(__file__))

def download_dataset(name):
	root_path = os.path.join(curr_path, name)
	if not os.path.isdir(root_path):
		os.mkdir(root_path)

	data_path = os.path.join(root_path, 'data')
	if not os.path.isdir(data_path):
		os.mkdir(data_path)

	labels_file = os.path.join(root_path, 'labels.txt')

	return root_path, data_path, labels_file

def download_cats_and_dogs(verbose=False):
	root_path, data_path, labels_file = download_dataset('cats_and_dogs')

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
				img = cv2.resize(img, (192, 192))
				cv2.imwrite(dest_file, img)
				os.remove(src_file)

		os.rmdir(os.path.join(root_path, 'train'))
		if os.path.isfile(os.path.join(root_path, 'train.zip')):
			os.remove(os.path.join(root_path, 'train.zip'))
	
	return data_path, labels_file

def cats_and_dogs(verbose=False):
	data_path, labels_file = download_cats_and_dogs(verbose=verbose)
	dataset = Dataset(root_folder=data_path, labels_file=labels_file, verbose=verbose)
	return dataset