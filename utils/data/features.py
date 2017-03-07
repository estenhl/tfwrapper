import os
import numpy as np

def parse_features(src, filename_col=0, label_col=1, features_start_col=2, delimiter=','):
	all_features = []

	if os.path.isfile(src):
		with open(src, 'r') as f:
			for line in f.readlines():
				tokens = line.split(delimiter)

				filename = tokens[filename_col]
				label = tokens[label_col]
				features = [float(x) for x in tokens[features_start_col:]]

				all_features.append({'filename': filename, 'label': label, 'features': features})

	return all_features

def write_features(dest, all_features):
	with open(dest, 'w') as f:
		for features in all_features:
			f.write(features['filename'] + ',' + features['label'] + ',' + ','.join([str(x) for x in features['features']]) + '\n')

def translate_features(all_features):
	X = []
	y = []

	for features in all_features:
		X.append(features['features'])
		y.append(features['label'])

	return np.asarray(X), y
