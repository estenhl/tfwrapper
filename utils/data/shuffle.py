import numpy as np

def shuffle(X, y):
	idx = np.arange(len(X))
	np.random.shuffle(idx)

	return np.squeeze(X[idx]), np.squeeze(y[idx])