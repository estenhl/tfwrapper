import os
import numpy as np
import pandas as pd

from tfwrapper.utils.exceptions import InvalidArgumentException

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

def is_list_of_dicts(obj):
	return type(obj) == list and (len(obj) == 0 or type(obj[0]) == dict)

def write_features(dest, all_features, delimiter='|', append=False, mode='w'):
	if append:
		mode = 'a'
	write_header = mode == 'w'

	if is_list_of_dicts(all_features):
		all_features = pd.DataFrame(all_features)
	elif type(all_features) != pd.DataFrame:
		raise InvalidArgumentException('Write features requires either a DataFrame or a list of dicts, not %s' % type(all_features))

	all_features['features_as_str'] = all_features['features'].apply(lambda x: delimiter.join([str(y) for y in x]))
	all_features[['filename', 'label', 'features_as_str']].to_csv(dest, mode=mode, sep=delimiter, index=False, header=write_header)