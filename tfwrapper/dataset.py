import os
import cv2
import numpy as np
from collections import Counter

from tfwrapper.utils.data import parse_features

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

	return X, y

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

	return X, y

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
			elif verbose:
				print('Skipping filename ' + src_file)

	return X, y

def parse_tokens_file(filename):
	tokens = []

	with open(filename, 'r') as f:
		for line in f.readlines():
			tokens += [x.strip() for x in line.split(' ') if len(x.strip()) > 0]

	return tokens

def tokens_to_indexes(tokens, add_none=True):
	tokenized_sequence = []
	indexes = []
	tokens_dict = {}

	if add_none:
		indexes.append(None)
		tokens_dict[None] = 0

	for token in tokens:
		if token not in tokens_dict:
			tokens_dict[token] = len(indexes)
			indexes.append(token)
		tokenized_sequence.append(tokens_dict[token])

	return tokenized_sequence, indexes, tokens_dict
	
class Dataset():
	X = []
	y = []

	def __init__(self, X=None, y=None, features=None, features_file=None, verbose=False):
		if features_file is not None:
			self.X, self.y = translate_features(parse_features(features_file))

		if features is not None:
			self.X, self.y = translate_features(features)

		if X is not None:
			self.X = X

		if y is not None:
			self.y = y

	def getdata(self, X=None, y=None, normalize=False, balance=False, translate_labels=False, shuffle=False, onehot=False, split=False):
		if X is None or y is None:
			X, y = self.X, self.y
		X, y = np.asarray(X), np.asarray(y)

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

		test_X, test_y = None, None

		if split:
			X, y, test_X, test_y = split_dataset(X, y)

		return X, y, test_X, test_y, labels

class ImageDataset(Dataset):
	def __init__(self, root_folder=None, labels_file=None, verbose=False):
		X, y = None, None

		if labels_file is not None and root_folder is not None:
			X, y = parse_folder_with_labels_file(root_folder, labels_file, verbose=verbose)
		elif root_folder is not None:
			X, y = parse_datastructure(root_folder, verbose=verbose)

		super().__init__(X=X, y=y, verbose=verbose)

class TokensDataset(Dataset):
	tokens = []
	indexes = []
	tokens_dict = {}

	def __init__(self, tokens=None, tokens_file=None):
		if tokens_file is not None:
			tokens = parse_tokens_file(tokens_file)

		if tokens is not None:
			self.tokens, self.indexes, self.tokens_dict = tokens_to_indexes(tokens)

	def token_to_index(self, token):
		return self.tokens_dict[token]

	def index_to_token(self, index):
		return self.indexes[index]

	def tokens_to_indexes(self, tokens):
		indexes = []

		for token in tokens:
			indexes.append(self.token_to_index(token))

		return np.asarray(indexes)

	def indexes_to_tokens(self, indexes):
		tokens = []

		for index in indexes:
			tokens.append(self.index_to_token(index))

		return np.asarray(tokens)

	def getdata(self, sequence_length, onehot=False, shuffle=False, split=False):
		X = []
		y = []

		for i in range(sequence_length):
			x = [self.token_to_index(None)] * ((sequence_length - 1) - i) + self.tokens[:i + 1]
			X.append(x)
			y.append(self.tokens[i + 1])

		for i in range(len(self.tokens) - sequence_length):
			X.append(self.tokens[i:i + sequence_length])
			y.append(self.tokens[i + sequence_length])

		return super().getdata(X=X, y=y, onehot=onehot, shuffle=shuffle, split=split)





