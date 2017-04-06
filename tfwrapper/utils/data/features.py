import os
import numpy as np
import pandas as pd

def parse_features_old(src, filename_col=0, label_col=1, features_start_col=2, delimiter='|'):
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

def write_features_old(dest, all_features, delimiter='|'):
	with open(dest, 'w') as f:
		for features in all_features:
			f.write(features['filename'] + delimiter + features['label'] + delimiter + delimiter.join([str(x) for x in features['features']]) + '\n')

def parse_features(src, filename_col=0, label_col=1, features_col=2, delimiter='|'):
	if os.path.isfile(src):
		features = pd.read_csv(src,
							   sep=delimiter,
							   header=0,
							   names=['filename', 'label', 'features_as_str'],
							   usecols=[filename_col, label_col, features_col])
		features['features'] = features['features_as_str'].apply(lambda x: np.fromstring(x, sep=delimiter))
		return features[['filename', 'label', 'features']]
	else:
		return pd.DataFrame(columns = ['filename', 'label', 'features'])

def write_features(dest, all_features, delimiter='|'):
	all_features['features_as_str'] = all_features['features'].apply(lambda x: delimiter.join([str(y) for y in x]))
	all_features[['filename', 'label', 'features_as_str']].to_csv(dest, sep=delimiter, index=False)