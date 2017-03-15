import os
import cv2
import numpy as np
from collections import Counter

def normalize_array(arr):
	return (arr - arr.mean()) / arr.std()

def shuffle_dataset(X, y):
	idx = np.arange(len(X))
	np.random.shuffle(idx)

	return np.squeeze(X[idx]), np.squeeze(y[idx])

def balance_dataset(X, y):
	assert len(X) == len(y)

	counts = Counter(y)
	min_count = min([counts[x] for x in counts])

	counters = {}
	for val in y:
		counters[val] = 0

	balanced_X = []
	balanced_y = []

	for i in range(0, len(X)):
		if counters[y[i]] < min_count:
			balanced_X.append(X[i])
			balanced_y.append(y[i])
		counters[y[i]] = counters[y[i]] + 1

	return np.asarray(balanced_X), np.asarray(balanced_y)

def onehot_array(arr):
	shape = (len(arr), np.amax(arr) + 1)
	onehot = np.zeros(shape)
	onehot[np.arange(shape[0]), arr] = 1

	return onehot

def translate_features(all_features):
	X = []
	y = []

	for features in all_features:
		X.append(features['features'])
		y.append(features['label'])

	return np.asarray(X), y

def labels_to_indexes(y):
	labels = []
	indices = []

	for label in y:
		if label not in labels:
			labels.append(label)
		indices.append(labels.index(label))

	return np.asarray(indices), labels

def split_dataset(X, y, val_split=0.8):
	train_len = int(len(X) * val_split)
	train_X = X[:train_len]
	train_y = y[:train_len]
	val_X = X[train_len:]
	val_y = y[train_len:]

	return train_X, train_y, val_X, val_y

def parse_datastructure(root, suffix='.jpg', verbose=False):
	X = []
	y = []

	for foldername in os.listdir(root):
		src = os.path.join(root, foldername)
		if os.path.isdir(src):
			for filename in os.listdir(src):
				if filename.endswith(suffix):
					src_file = os.path.join(src, filename)
					img = cv2.imread(src_file)
					X.append(img)
					y.append(foldername)
				elif verbose:
					print('Skipping filename ' + filename)
		elif verbose:
			print('Skipping foldername ' + foldername)

	return np.asarray(X), np.asarray(y)

def parse_folder_with_labels_file(root, labels_file, verbose=False):
	X = []
	y = []

	with open(labels_file, 'r') as f:
		for line in f.readlines():
			label, filename = line.split(',')
			src_file = os.path.join(root, filename).strip()
			if os.path.isfile(src_file):
				img = cv2.imread(src_file)
				X.append(img)
				y.append(label)
			else:
				print('Skipping filename ' + src_file)

	return np.asarray(X), np.asarray(y)

class Dataset():
	X = np.asarray([])
	y = np.asarray([])

	def __init__(self, X=None, y=None, features=None, root_folder=None, labels_file=None, verbose=False):
		if labels_file is not None and root_folder is not None:
			self.X, self.y = parse_folder_with_labels_file(root_folder, labels_file, verbose=verbose)
		elif root_folder is not None:
			self.X, self.y = parse_datastructure(root_folder, verbose=verbose)

		if features is not None:
			self.X, self.y = translate_features(features)

		if X is not None:
			self.X = X

		if y is not None:
			self.y = y

	def getdata(self, normalize=False, balance=False, translate_labels=False, shuffle=False, onehot=False, split=False):
		X = self.X
		y = self.y
		print('X.shape: ' + str(X.shape))
		print('y.shape: ' + str(y.shape))
		labels = []

		if normalize:
			X = normalize_array(X)

		if translate_labels:
			y, labels = labels_to_indexes(y)

		if shuffle:
			X, y = shuffle_dataset(X, y)

		if balance:
			X, y = balance_dataset(X, y)

		if onehot:
			y = onehot_array(y)

		if split:
			X, y, test_X, test_y = split_dataset(X, y)
			return X, y, test_X, test_y, labels
		else:
			return X, y, labels