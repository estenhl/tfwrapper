import os
import pytest
import numpy as np

from tfwrapper.utils.data import parse_features
from tfwrapper.utils.data import write_features

from utils import generate_features

def test_parse_written_features():
	_, _, features = generate_features()
	write_features('tmp.csv', features)
	parsed_features = parse_features('tmp.csv')
	os.remove('tmp.csv')

	assert len(features) == len(parsed_features)

	features = sorted(features, key=lambda x: x['filename'])
	parsed_features = sorted(parsed_features, key=lambda x: x['filename'])

	for i in range(len(features)):
		assert features[i]['filename'] == parsed_features[i]['filename']
		assert features[i]['label'] == parsed_features[i]['label']
		assert np.array_equal(features[i]['features'], parsed_features[i]['features'])


