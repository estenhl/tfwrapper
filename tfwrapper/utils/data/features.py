import os
import numpy as np
import pandas as pd


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