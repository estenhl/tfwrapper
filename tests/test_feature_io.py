import os
import pytest
import numpy as np

from tfwrapper.utils.data import parse_features
from tfwrapper.utils.data import write_features

from .utils import generate_features

def test_parse_written_features():
	_, _, features = generate_features()
	write_features('tmp.csv', features)
	parsed_features = parse_features('tmp.csv')
	os.remove('tmp.csv')

	assert len(features) == len(parsed_features)

	parsed_features = parsed_features.sort_values(by=['filename'])

	for i in range(len(features)):
		assert features.ix[i, 'filename'] == parsed_features.ix[i, 'filename']
		assert features.ix[i, 'label'] == parsed_features.ix[i, 'label']
		assert np.array_equal(features.ix[i, 'features'], parsed_features.ix[i, 'features'])


