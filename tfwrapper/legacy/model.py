import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

DEFAULT_VAL_SPLIT = 0.8

class Model(ABC):
	def split_data(self, X, y, val_split=DEFAULT_VAL_SPLIT):
		train_len = int(len(X) * val_split)
		train_X = X[:train_len]
		train_y = y[:train_len]
		val_X = X[train_len:]
		val_y = y[train_len:]

		return train_X, train_y, val_X, val_y
		
	def batch_data(self, batch_size, X, y=None):
		batches = []

		for i in range(0, int(len(X) / batch_size) + 1):
			start = (i * batch_size)
			end = min((i + 1) * batch_size, len(X))
			batch_X = X[start:end]
			if y is None:
				batches.append({'x': batch_X})
			else:
				batch_y = y[start:end]
				batches.append({'x': batch_X, 'y': batch_y})

		return batches