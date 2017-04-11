import os
import cv2
import scipy.ndimage
import numpy as np
from collections import Counter
from random import shuffle
from tfwrapper.utils.data import parse_features

def normalize_array(arr):
	return (arr - arr.mean()) / arr.std()

def shuffle_dataset(X, y):
	idx = np.arange(len(X))
	np.random.shuffle(idx)

	return np.squeeze(X[idx]), np.squeeze(y[idx])

def balance_dataset(X, y):
	assert len(X) == len(y)

	is_onehot = False
	if len(y.shape) > 1 and y.shape[-1] > 1:
		is_onehot = True
		print('TRANSLATING')
		y = [np.argmax(y_) for y_ in y]

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

	if is_onehot:
		y = onehot(y)

	return np.asarray(balanced_X), np.asarray(balanced_y)

def onehot_array(arr):
	shape = (len(arr), np.amax(arr) + 1)
	onehot = np.zeros(shape)
	for i in range(len(arr)):
		onehot[i][arr[i]] = 1

	return np.asarray(onehot)

def labels_to_indexes(y, sort=True):
	labels = []
	indices = []

	if sort:
		for label in y:
			if label not in labels:
				labels.append(label)
		labels = sorted(labels)

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

	for foldername in os.listdir(root):
		src = os.path.join(root, foldername)
		if os.path.isdir(src):
			for filename in os.listdir(src):
				if filename.endswith(suffix):
					src_file = os.path.join(src, filename)
					X.append(src_file)
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
				X.append(src_file)
				y.append(label)
			elif verbose:
				print('Skipping filename ' + src_file)

	return np.asarray(X), np.asarray(y)

def drop_classes(X, y, *, keep):
	filtered_X = []
	filtered_y = []

	for i in range(len(X)):
		if y[i] in keep:
			filtered_X.append(X[i])
			filtered_y.append(y[i])

	return np.asarray(filtered_X), np.asarray(filtered_y)

class Dataset():

	@property
	def X(self):
		return self._X

	@property
	def y(self):
		return self._y

	def __init__(self, X=np.asarray([]), y=np.asarray([]), labels=np.asarray([]), features=None, features_file=None, verbose=False):
		self._X = X
		self._y = y
		self.labels = labels

		if features_file is not None:
			parsed_features = parse_features(features_file)
			self._X = np.asarray(parsed_features['features'].tolist())
			self._y = np.asarray(parsed_features['label'].tolist())

		if features is not None:
			self._X = np.asarray(features['features'].tolist())
			self._y = np.asarray(features['label'].tolist())

	def normalize(self):
		return self.__class__(X=normalize_array(self._X), y=self._y, labels=self.labels)

	def shuffle(self):
		X, y = shuffle_dataset(self._X, self._y)

		return self.__class__(X=X, y=y, labels=self.labels)

	def balance(self):
		X, y = balance_dataset(self._X, self._y)

		return self.__class__(X=X, y=y, labels=self.labels)

	def translate_labels(self):
		y, labels = labels_to_indexes(self._y)

		return self.__class__(X=self._X, y=y, labels=labels)

	def onehot(self):
		return self.__class__(X=self._X, y=onehot_array(self._y), labels=self.labels)

	def split(self, ratio=0.8):
		X, y, test_X, test_y = split_dataset(self._X, self._y, ratio=ratio)
		train_dataset = self.__class__(X=X, y=y, labels=self.labels)
		test_dataset = self.__class__(X=test_X, y=test_y, labels=self.labels)

		return train_dataset, test_dataset

	def drop_classes(self, *, drop=None, keep=None):
		if drop is None and keep is None:
			raise InvalidArgumentException('When dropping classes, either a list of classes to drop or a list of classes to keep must be supplied')
		elif drop is None:
			_keep = set(self._y) & set(keep)
		else:
			_keep = set(self._y) - set(drop)

		X, y = drop_classes(self._X, self._y, keep=_keep)

		return self.__class__(X=X, y=y, labels=self.labels)

	def __len__(self):
		return len(self._X)

from tfwrapper import twimage

RESIZE = "resize"
TRANSFORM_BW = "bw"
FLIP_LR = "fliplr"
FLIP_UD = "flipud"

def create_name(name, suffixes):
    img_part = name.rsplit(".", 1)

    suffix_string = "_".join(suffixes)

    return "{}_{}.{}".format(img_part[0], suffix_string, img_part[1])


class ImagePreprocess():
    def __init__(self):
        self.augs = {}

    # (width, heiht)
    def resize(self, img_size=(299, 299)):
        print('SAT RESIZE IN ' + str(self))
        self.augs[RESIZE] = img_size

    def bw(self):
        self.augs[TRANSFORM_BW] = True

    def append_flip_lr(self):
        self.augs[FLIP_LR] = True

    def append_flip_ud(self):
        self.augs[FLIP_UD] = True

    def apply_file(self, image_path, name):
        img = twimage.imread(image_path)
        return self.apply(img, name)

    def apply(self, img, name):
        img_versions = []
        img_names = []

        org_suffixes = []

        if RESIZE in self.augs:
            img = twimage.resize(img, self.augs[RESIZE])
            #Should check for size
            org_suffixes.append(RESIZE)
        if TRANSFORM_BW in self.augs:
            img = twimage.bw(img, shape=3)
            org_suffixes.append(TRANSFORM_BW)

        img_versions.append(img)
        img_names.append(create_name(name, org_suffixes))

        # img_versions.append(img)
        # img_names.append(org_suffixes)
        # #Append
        if FLIP_LR in self.augs:
            img_versions.append(np.fliplr(img))
            org_suffixes.append(FLIP_LR)
            img_names.append(create_name(name, org_suffixes))
            org_suffixes.remove(FLIP_LR)

        if FLIP_UD in self.augs:
            img_versions.append(np.flipud(img))
            org_suffixes.append(FLIP_UD)
            img_names.append(create_name(name, org_suffixes))
            org_suffixes.remove(FLIP_UD)

        return img_names, img_versions

class ImageDataset(Dataset):
	preprocessor = ImagePreprocess()

	@property
	def X(self):
		X, _, _ = self.read_batch(0)
		return X

	@property
	def y(self):
		_, y, _ = self.read_batch(0)
		return y

	def __init__(self, X=np.asarray([]), y=np.asarray([]), labels=np.asarray([]), root_folder=None, labels_file=None):
		_X = X
		_y = y

		if labels_file is not None and root_folder is not None:
			_X, _y = parse_folder_with_labels_file(root_folder, labels_file)
		elif root_folder is not None:
			_X, _y = parse_datastructure(root_folder)

		self.cursor = 0
		super().__init__(X=_X, y=_y, labels=labels)

	def next_batch(self, batch_size=128):
		while self.cursor + batch_size < len(self):
			batch_X, batch_y, cursor = self.read_batch(self.cursor)
			self.cursor = cursor
			yield Dataset(X=batch_X, y=batch_y, labels=self.labels)

	def read_batch(self, cursor, batch_size=float('inf')):
		batch_X = []
		batch_y = []

		while len(batch_X) < batch_size and cursor < len(self):
			_, imgs = self.preprocessor.apply_file(self._X[cursor], 'Fucknames.jpg')
			batch_X += imgs
			batch_y += [self._y[cursor]] * len(imgs)
			cursor += 1

		X = np.asarray(batch_X)
		y = np.asarray(batch_y)

		return X, y, cursor

# class ImageTransformer():
# 	def __init__(self, resize_to=None, bw=False, hflip=False, vflip=False, rotation_steps=0, max_rotation_angle=0, blur_steps=0, max_blur_sigma=0):
# 		self.resize_to = resize_to
# 		self.bw = bw
# 		self.hflip = hflip
# 		self.vflip = vflip
# 		self.rotation_steps = rotation_steps
# 		self.max_rotation_angle = max_rotation_angle
# 		self.blur_steps = blur_steps
# 		self.max_blur_sigma = max_blur_sigma

# 	def transform(self, img):
# 		img_variants = []
# 		suffixes = ['']

# 		if self.resize_to is not None:
# 			img = cv2.resize(img, self.resize_to)

# 		if self.bw:
# 			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 		img_variants.append(img)

# 		if self.hflip:
# 			img_variants.append(np.fliplr(img))
# 			suffixes.append('_hflip')

# 		if self.vflip:
# 			img_variants.append(np.flipud(img))
# 			suffixes.append('_vflip')

# 		if self.hflip and self.vflip:
# 			img_variants.append(np.fliplr(np.flipud(img)))
# 			suffixes.append('_hvflip')

# 		if self.rotation_steps > 0 and self.max_rotation_angle > 0:
# 			for i in range(self.rotation_steps):
# 				angle = self.max_rotation_angle * (i+1)/self.rotation_steps
# 				img_variants.append(scipy.ndimage.interpolation.rotate(img, angle, reshape=False))
# 				suffixes.append('_rotate_' + str(angle))
# 				img_variants.append(scipy.ndimage.interpolation.rotate(img, -angle, reshape=False))
# 				suffixes.append('_rotate_' + str(-angle))

# 		if self.blur_steps > 0 and self.max_blur_sigma > 0:
# 			for i in range(self.blur_steps):
# 				sigma = self.max_blur_sigma * (i+1)/self.blur_steps
# 				img_variants.append(scipy.ndimage.filters.gaussian_filter(img, sigma))
# 				suffixes.append('_blur_' + str(sigma))

# 		# TODO: generate combinations of flip, rotation and blur

# 		return img_variants, suffixes
