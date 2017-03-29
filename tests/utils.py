import os
import numpy as np

curr_path = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))

def generate_features():
	X = np.asarray([
		[0.01, 0.02, 0.03],
		[0.04, 0.05, 0.06],
		[0.07, 0.08, 0.09]
	])
	y = np.asarray([
		'label1',
		'label2',
		'label3'
	])
	features = [
		{'filename': 'file1', 'label': y[0], 'features': X[0]},
		{'filename': 'file2', 'label': y[1], 'features': X[1]},
		{'filename': 'file3', 'label': y[2], 'features': X[2]}
	]

	return X, y, features