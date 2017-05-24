import numpy as np

from tfwrapper.ensembles.utils import accumulate_predictions
from tfwrapper.ensembles.utils import vote_predictions

def generate_preds():
    return np.asarray([
        [
            [0.0, 1.0],
            [1.3, 2.3],
            [2.7, 0.7]
        ],
        [
            [2.3, 0.3],
            [1.7, 2.7],
            [0.0, 1.0]
        ],
        [
            [1.7, 0.7],
            [0.0, 2.0],
            [2.3, 1.3]
        ]
    ])

def test_accumulate_predictions():
    preds = generate_preds()

    expected = np.asarray([
        [4.0, 2.0],
        [3.0, 7.0],
        [5.0, 3.0]
    ])

    accumulated = accumulate_predictions(preds)

    assert np.array_equal(expected, accumulated)

def test_majority_vote_predictions():
    preds = generate_preds()

    expected = np.asarray([
        [2.0, 1.0],
        [0.0, 3.0],
        [2.0, 1.0]
    ])

    votes = vote_predictions(preds)

    assert np.array_equal(expected, votes)