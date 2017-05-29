import os
import numpy as np

import tfwrapper

print(tfwrapper.__version__)

from tfwrapper.dataset import parse_datastructure
from tfwrapper import Dataset
from tfwrapper.utils.files import write_features

from utils import curr_path
from utils import generate_features


def test_parse_datastructure():
    root_path = os.path.join(os.path.dirname(__file__), 'data/testset')
    X, y = parse_datastructure(root_path)

    assert X.shape[0] == 2
    assert X.shape[0] == 2


def test_create_from_data():
    X = np.asarray([1, 2, 3])
    y = np.asarray([2, 4, 6])
    dataset = Dataset(X=X, y=y)

    assert np.array_equal(X, dataset.X)
    assert np.array_equal(y, dataset.y)


def test_create_from_features():
    X, y, features = generate_features()
    dataset = Dataset(features=features)

    assert np.array_equal(X, dataset.X)
    assert np.array_equal(y, dataset.y)


def test_create_from_feature_file():
    X, y, features = generate_features()
    tmp_file = os.path.join(curr_path, 'tmp.csv')
    write_features(tmp_file, features)
    dataset = Dataset(features_file=tmp_file)
    os.remove(tmp_file)

    assert np.array_equal(X, dataset.X)
    assert np.array_equal(y, dataset.y)


def test_normalize():
    X = np.asarray([5, 4, 3])
    y = np.asarray([1, 1, 1])
    dataset = Dataset(X=X, y=y)
    dataset = dataset.normalize()

    assert 0 < dataset.X[0]
    assert 0 == dataset.X[1]
    assert 0 > dataset.X[2]
    assert dataset.X[0] == -dataset.X[2]
    assert np.array_equal(y, dataset.y)


def test_balance():
    X = np.zeros(100)
    y = np.concatenate([np.zeros(10), np.ones(90)])
    dataset = Dataset(X=X, y=y)
    dataset = dataset.balance()

    assert 20 == len(dataset.X)
    assert 20 == len(dataset.y)
    assert 10 == np.sum(dataset.y)


def test_balance_with_max():
    X = np.concatenate([np.zeros(5), np.ones(10)])
    y = np.concatenate([np.zeros(5), np.ones(10)])
    dataset = Dataset(X=X, y=y)
    dataset = dataset.balance(max=8)

    assert 5 + 8 == len(dataset)


def test_balance_with_low_max():
    X = np.concatenate([np.zeros(3), np.ones(3)])
    y = np.concatenate([np.zeros(3), np.ones(3)])
    dataset = Dataset(X=X, y=y)
    dataset = dataset.balance(max=2)

    assert 4 == len(dataset)


def test_translate_labels():
    X = np.asarray([0, 1, 2])
    y = np.asarray(['Zero', 'One', 'Two'])
    dataset = Dataset(X=X, y=y)
    dataset = dataset.translate_labels()
    labels = dataset.labels

    assert 3 == len(dataset.labels)
    assert 'Zero' == labels[dataset.y[np.where(dataset.X == 0)[0][0]]]
    assert 'One' == labels[dataset.y[np.where(dataset.X == 1)[0][0]]]
    assert 'Two' == labels[dataset.y[np.where(dataset.X == 2)[0][0]]]


def test_shuffle():
    X = np.concatenate([np.zeros(100), np.ones(100)])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    dataset = Dataset(X=X, y=y)
    dataset = dataset.shuffle()

    assert not np.array_equal(X, dataset.X)
    assert np.array_equal(dataset.X, dataset.y)


def test_onehot():
    X = np.zeros(10)
    y = np.arange(10)
    dataset = Dataset(X=X, y=y)
    dataset = dataset.onehot()

    assert (10, 10) == dataset.y.shape
    for i in range(10):
        arr = np.zeros(10)
        arr[i] = 1
        assert np.array_equal(arr, dataset.y[i])


def test_split():
    X = np.concatenate([np.zeros(80), np.ones(20)])
    y = np.concatenate([np.zeros(80), np.ones(20)])
    dataset = Dataset(X=X, y=y)
    train_dataset, test_dataset = dataset.split(ratio=0.8)

    assert np.array_equal(train_dataset.X, np.zeros(80))
    assert np.array_equal(train_dataset.y, np.zeros(80))
    assert np.array_equal(test_dataset.X, np.ones(20))
    assert np.array_equal(test_dataset.y, np.ones(20))


def test_length():
    X = np.zeros(100)
    y = np.zeros(100)
    dataset = Dataset(X=X, y=y)

    assert len(X) == len(dataset)


def test_drop_classes():
    X = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    dataset = Dataset(X=X, y=y)
    dataset = dataset.drop_classes(drop=[2, 3])

    assert 6 == len(dataset.X)
    assert 6 == len(dataset.y)


def test_keep_classes():
    X = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    dataset = Dataset(X=X, y=y)
    dataset = dataset.drop_classes(keep=[1, 2, 3])

    assert 9 == len(dataset.X)
    assert 9 == len(dataset.y)


def test_add_datasets():
    X1 = np.arange(10)
    y1 = np.arange(10)
    X2 = np.arange(10) + 10
    y2 = np.arange(10) + 10

    dataset1 = Dataset(X=X1, y=y1)
    dataset2 = Dataset(X=X2, y=y2)
    dataset = dataset1 + dataset2

    assert len(X1) + len(X2) == len(dataset)
    for i in range(20):
        assert i in dataset.X
        assert dataset.X[i] == dataset.y[i]


def test_slice_datasets():
    X = np.arange(10)
    y = np.arange(10)
    dataset = Dataset(X=X, y=y)
    dataset = dataset[:5]

    assert 5 == len(dataset)
    for i in range(5):
        assert i in dataset.X
        assert dataset.X[i] == dataset.y[i]


def test_index_datasets():
    X = np.arange(10)
    y = np.arange(10)
    dataset = Dataset(X=X, y=y)
    dataset = dataset[5]

    assert 1 == len(dataset)
    assert np.array_equal(np.asarray([5]), dataset.X)
    assert np.array_equal(np.asarray([5]), dataset.y)


def test_equal_folds():
    X = np.arange(12)
    y = np.arange(12)
    dataset = Dataset(X=X, y=y)

    k = 3
    folds = dataset.folds(k)

    assert k == len(folds)
    for i in range(k):
        for j in range(int(len(X) / k)):
            assert (i * int(len(X) / k)) + j in folds[i].X
            assert (i * int(len(X) / k)) + j in folds[i].y


def test_unequal_folds():
    X = np.arange(7)
    y = np.arange(7)
    dataset = Dataset(X=X, y=y)

    folds = dataset.folds(2)

    assert 2 == len(folds)
    assert 4 == len(folds[0])
    assert 3 == len(folds[1])


def test_merge_classes():
    X = np.arange(10)
    y = np.arange(10)
    dataset = Dataset(X=X, y=y)

    assert 10 == len(np.unique(dataset.y))
    assert 10 == len(np.unique(dataset.X))

    mappings = {}
    for i in range(10):
        mappings[i] = int(i / 5)
    dataset = dataset.merge_classes(mappings)

    assert 2 == len(np.unique(dataset.y))
    assert 10 == len(np.unique(dataset.X))

    for i in range(len(dataset)):
        assert dataset.y[i] == int(dataset.X[i] / 5)


def test_inherit_labels():
    X = np.arange(10)
    y = np.arange(10)
    labels = [str(x) for x in np.arange(10)]
    dataset = Dataset(X=X, y=y, labels=labels)
    dataset = dataset.onehot()

    assert np.array_equal(labels, dataset.labels)


def test_shape():
    X = np.reshape(np.arange(27), (3, 3, 3))
    y = np.reshape(np.arange(27), (3, 3, 3))
    dataset = Dataset(X=X, y=y)

    assert X.shape == dataset.shape


def test_invalid_arg_throws_exception():
    exception = False
    try:
        dataset = Dataset(x=None, y=None)
    except Exception:
        exception = True

    assert exception
