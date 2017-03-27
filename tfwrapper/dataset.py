import os
import cv2
import numpy as np
from collections import Counter

from tfwrapper.utils.data import parse_features

def normalize_array(arr):
	return (arr - arr.mean()) / arr.std()

def shuffle_dataset(X, y, names=None):
	idx = np.arange(len(X))
	np.random.shuffle(idx)

	if names is None:
		return np.squeeze(X[idx]), np.squeeze(y[idx])
	else:
		return np.squeeze(X[idx]), np.squeeze(y[idx]), np.squeeze(names[idx])

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
	for i in range(len(arr)):
		onehot[i][arr[i]] = 1

	return np.asarray(onehot)

def translate_features(all_features):
	X = []
	y = []

	for features in all_features:
		X.append(features['features'])
		y.append(features['label'])

	return np.asarray(X), np.asarray(y)

def labels_to_indexes(y):
	labels = []
	indices = []

	for label in y:
		if label not in labels:
			labels.append(label)
		indices.append(labels.index(label))

	return np.asarray(indices), np.asarray(labels)

def split_dataset(X, y, ratio=0.8):
	train_len = int(len(X) * ratio)
	train_X = X[:train_len]
	train_y = y[:train_len]
	test_X = X[train_len:]
	test_y = y[train_len:]

	return np.asarray(train_X), np.asarray(train_y), np.asarray(test_X), np.asarray(test_y)

def parse_datastructure(root, suffix='.jpg', verbose=False):
	X = []
	y = []
	names = []

	for foldername in os.listdir(root):
		src = os.path.join(root, foldername)
		if os.path.isdir(src):
			for filename in os.listdir(src):
				if filename.endswith(suffix):
					src_file = os.path.join(src, filename)
					img = cv2.imread(src_file)
					X.append(img)
					y.append(foldername)
					names.append(filename)
				elif verbose:
					print('Skipping filename ' + filename)
		elif verbose:
			print('Skipping foldername ' + foldername)

	return np.asarray(X), np.asarray(y), np.asarray(names)

def parse_folder_with_labels_file(root, labels_file, verbose=False):
	X = []
	y = []
	names = []

	with open(labels_file, 'r') as f:
		for line in f.readlines():
			label, filename = line.split(',')
			src_file = os.path.join(root, filename).strip()
			if os.path.isfile(src_file):
				img = cv2.imread(src_file)
				X.append(img)
				y.append(label)
				names.append(filename)
			elif verbose:
				print('Skipping filename ' + src_file)

	return np.asarray(X), np.asarray(y), np.asarray(names)

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

	return np.asarray(tokenized_sequence), np.asarray(indexes), np.asarray(tokens_dict)
	
class Dataset():
	def __init__(self, X=np.asarray([]), y=np.asarray([]), labels=np.asarray([]), features=None, features_file=None, verbose=False):
		self.X = X
		self.y = y
		self.labels = labels

		if features_file is not None:
			self.X, self.y = translate_features(parse_features(features_file))

		if features is not None:
			self.X, self.y = translate_features(features)

	def normalize(self):
		return Dataset(X=normalize_array(self.X), y=self.y, labels=self.labels)

	def shuffle(self):
		X, y = shuffle_dataset(self.X, self.y)

		return Dataset(X=X, y=y, labels=self.labels)

	def balance(self):
		X, y = balance_dataset(self.X, self.y)

		return Dataset(X=X, y=y, labels=self.labels)

	def translate_labels(self):
		y, labels = labels_to_indexes(self.y)

		return Dataset(X=self.X, y=y, labels=labels)

	def onehot(self):
		return Dataset(X=self.X, y=onehot_array(self.y), labels=self.labels)

	def split(self, ratio=0.8):
		X, y, test_X, test_y = split_dataset(self.X, self.y, ratio=ratio)
		train_dataset = Dataset(X=X, y=y, labels=self.labels)
		test_dataset = Dataset(X=test_X, y=test_y, labels=self.labels)

		return train_dataset, test_dataset

class ImageTransformer():
	resize_to = None
	black_and_white = False


class ImageDataset(Dataset):
	names = None

	def __init__(self, X=None, y=None, names=None, root_folder=None, labels_file=None, verbose=False):
		if labels_file is not None and root_folder is not None:
			X, y, names = parse_folder_with_labels_file(root_folder, labels_file, verbose=verbose)
		elif root_folder is not None:
			X, y, names = parse_datastructure(root_folder, verbose=verbose)

		super().__init__(X=X, y=y, verbose=verbose)
		self.names = names

	def getdata(self, normalize=False, balance=False, translate_labels=False, 
				shuffle=False, onehot=False, split=False, transformer=None):
		if transformer:
			X = []
			y = []
			names = []

			for i in range(len(self.X)):
				variants, suffixes = transformer.transform(self.X[i])
				X += variants
				y += [self.y[i]] * len(variants)

				basename = self.names[i]
				if len(basename.split('.')) > 2:
					raise NotImplementedError('Filenames with . not allowed')

				prefix, filetype = basename.split('.')
				names += [prefix + suffix + '.' + filetype for suffix in suffixes]

			X = np.asarray(X)
			y = np.asarray(y)
			names = np.asarray(names)
		else:
			X = np.asarray(self.X)
			y = np.asarray(self.y)
			names = np.asarray(self.names)


		if shuffle:
			X, y, names = shuffle_dataset(X, y, names)


		X, y, test_X, test_y, labels = super().getdata(X=X, y=y, normalize=normalize, balance=balance, 
			translate_labels=translate_labels, onehot=onehot, split=split)
		return X, y, test_X, test_y, labels, names


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
