import os
import pytest
import random
import numpy as np
import pandas as pd

from tfwrapper.utils.data import parse_features
from tfwrapper.utils.data import write_features
from tfwrapper.utils.exceptions import InvalidArgumentException

from utils import curr_path
from utils import generate_features

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

def test_no_label():
    filename = os.path.join(curr_path, 'test_no_label.tmp')

    df = pd.DataFrame(columns=['filename', 'label', 'features'])
    for i in range(3):
        df.append({'filename': 'file%d' % i, 'features': np.arange(i+1, (i+1)*10).astype(float)}, ignore_index=True)

    write_features(filename, df)
    features = parse_features(filename)
    
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
    filename = os.path.join(curr_path, 'test_parse_written.tmp')
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

def test_parse_legacy_format():
    filename = os.path.join(curr_path, 'test_parse_legacy.tmp')

    num_cases = 3
    num_features = 2048
    with open(filename, 'w') as f:
        f.write('filename|label|features_as_str\n')
        for i in range(num_cases):
            f.write('case%d|%d|"' % (i, i))
            features = []
            for j in range(num_features):
                features.append('%.7f' % random.uniform(0, 1))
            f.write('%s"\n' % '|'.join(features))

    features = parse_features(filename)
    os.remove(filename)

    assert num_cases == len(features), 'Unable to parse legacy feature format'
    assert num_features == len(np.asarray(features.ix[0, 'features'])), 'Unable to parse features in legacy feature format'

def test_parse_new_format():
    filename = os.path.join(curr_path, 'test_parse_new.tmp')

    num_cases = 3
    num_features = 2048
    with open(filename, 'w') as f:
        f.write('filename|label|features_as_str\n')
        for i in range(num_cases):
            f.write('case%d|%d|"' % (i, i))
            features = []
            for j in range(num_features):
                features.append(float('%.7f' % random.uniform(0, 1)))
            f.write('%s"\n' % str(features))

    features = parse_features(filename)
    os.remove(filename)

    assert num_cases == len(features), 'Unable to parse new feature format'
    assert num_features == len(np.asarray(features.ix[0, 'features'])), 'Unable to parse features in new feature format'

def test_parse_complex_format():
    filename = os.path.join(curr_path, 'test_parse_complex.tmp')

    num_cases = 3
    height = 73
    width = 73
    depth = 64
    with open(filename, 'w') as f:
        f.write('filename|label|features_as_str\n')
        for i in range(num_cases):
            f.write('case%d|%d|"' % (i, i))
            features = np.zeros((height, width, depth))
            for j in range(height):
                for k in range(width):
                    for l in range(depth):
                        features[j][k][l] = float('%.7f' % random.uniform(0, 1))
            f.write('%s"\n' % str(features.tolist()))

    features = parse_features(filename)
    os.remove(filename)

    assert num_cases == len(features), 'Unable to parse new feature format with complex features'
    assert (height, width, depth) == np.asarray(features.ix[0, 'features']).shape, 'Unable to parse complex features in new feature format'

def test_write_complex_format():
    filename = os.path.join(curr_path, 'test_write_complex.tmp')

    data = []
    num_cases = 3
    height = 73
    width = 73
    depth = 64

    for i in range(num_cases):
        features = np.zeros((height, width, depth))
        for j in range(height):
            for k in range(width):
                for l in range(depth):
                    features[j][k][l] = float('%.7f' % random.uniform(0, 1))
        data.append({'filename': 'file%d' % i, 'label': 'label%d' % i, 'features': features})

    write_features(filename, data)

    with open(filename, 'r') as f:
        lines = f.readlines()
    os.remove(filename)

    assert num_cases + 1 == len(lines), 'Unable to write complex features in new feature format'

def test_parse_written_complex():
    filename = os.path.join(curr_path, 'test_parse_written_complex.tmp')

    data = []
    num_cases = 3
    height = 73
    width = 73
    depth = 64

    for i in range(num_cases):
        features = np.zeros((height, width, depth))
        for j in range(height):
            for k in range(width):
                for l in range(depth):
                    features[j][k][l] = float('%.7f' % random.uniform(0, 1))
        data.append({'filename': 'file%d' % i, 'label': 'label%d' % i, 'features': features})

    write_features(filename, data)
    features = parse_features(filename)
    os.remove(filename)

    assert num_cases == len(features), 'Unable to parse written features on new feature format'
    for i in range(num_cases):
        assert data[i]['filename'] == str(features.ix[i, 'filename']), 'Unable to parse filenames from written features on new format'
        assert data[i]['label'] == str(features.ix[i, 'label']), 'Unable to parse labels from written features on new format'
        assert np.array_equal(data[i]['features'], np.asarray(features.ix[i, 'features'])), 'Unable to parse complex features from written features on new format'
