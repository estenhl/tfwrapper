import os
import pytest
import numpy as np
import pandas as pd

from tfwrapper.utils.data import parse_features
from tfwrapper.utils.data import write_features
from tfwrapper.utils.exceptions import InvalidArgumentException

from .utils import curr_path
from .utils import generate_features

def test_write_new_file():
	filename = os.path.join(curr_path, 'test_write_new_file.tmp')
	assert not os.path.isfile(filename)

	df = pd.DataFrame(columns = ['filename', 'label', 'features'])
	records = []
	for i in range(3):
		records.append({'filename': 'file%d' % i, 
						'label': 'label%d' % i, 
						'features': np.arange(i, i*10).astype(float)})

	df = df.append(records, ignore_index=True)
	write_features(filename, df)

	assert os.path.isfile(filename)
	with open(filename, 'r') as f:
		assert len(records) + 1 == len(f.readlines())

	os.remove(filename)

def test_write_append():
	filename = os.path.join(curr_path, 'test_write_append.tmp')
	assert not os.path.isfile(filename)

	records = []
	for i in range(3):
		records.append({'filename': 'file%d' % i, 
						'label': 'label%d' % i, 
						'features': np.arange(i, i*10).astype(float)})

	write_features(filename, records[:1])
	write_features(filename, records[1:2], append=True)
	write_features(filename, records[2:], append=True)

	assert os.path.isfile(filename)
	with open(filename, 'r') as f:
		assert len(records) + 1 == len(f.readlines())

	os.remove(filename)

def test_change_mode():
	filename = os.path.join(curr_path, 'test_change_mode.tmp')
	assert not os.path.isfile(filename)

	records = []
	for i in range(3):
		records.append({'filename': 'file%d' % i, 
						'label': 'label%d' % i, 
						'features': np.arange(i, i*10).astype(float)})

	write_features(filename, records[:1])
	write_features(filename, records[1:2], mode='a')
	write_features(filename, records[2:], mode='a')

	assert os.path.isfile(filename)
	with open(filename, 'r') as f:
		assert len(records) + 1 == len(f.readlines())

	os.remove(filename)

def test_write_list_of_dicts():
	filename = os.path.join(curr_path, 'test_write_list_of_dicts.tmp')
	assert not os.path.isfile(filename)

	records = []
	for i in range(3):
		records.append({'filename': 'file%d' % i, 
						'label': 'label%d' % i, 
						'features': np.arange(i, i*10).astype(float)})

	write_features(filename, records)

	assert os.path.isfile(filename)
	with open(filename, 'r') as f:
		assert len(records) + 1 == len(f.readlines())

	os.remove(filename)

def test_valid_datatypes():
	filename = os.path.join(curr_path, 'test_valid_datatypes.tmp')

	valid_datatypes = [
				pd.DataFrame(columns = ['filename', 'label', 'features']),
				[{'filename': 'f', 'label': 'l', 'features': np.arange(5).astype(float)}],
				]

	exception = False
	for datatype in valid_datatypes:
		try:
			write_features(filename, datatype)
		except InvalidArgumentException:
			exception = True
		assert False == exception

	invalid_datatypes = ['Hei', 0, np.arange(5), [1, 2, 3], {'a': 1}]

	exception = False
	for datatype in invalid_datatypes:
		try:
			write_features(filename, datatype)
		except InvalidArgumentException:
			exception = True
		assert True == exception

	os.remove(filename)

def test_parse_written_features():
	filename = filename = os.path.join(curr_path, 'test_parse_written.tmp')
	_, _, features = generate_features()
	write_features(filename, features)
	parsed_features = parse_features(filename)
	os.remove(filename)

	assert len(features) == len(parsed_features)

	parsed_features = parsed_features.sort_values(by=['filename'])

	for i in range(len(features)):
		assert features.ix[i, 'filename'] == parsed_features.ix[i, 'filename']
		assert features.ix[i, 'label'] == parsed_features.ix[i, 'label']
		assert np.array_equal(features.ix[i, 'features'], parsed_features.ix[i, 'features'])


