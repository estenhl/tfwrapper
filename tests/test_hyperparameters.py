import pytest

from tfwrapper.hyperparameters import adjust_at_epochs
from tfwrapper.hyperparameters import adjust_after_epoch

def test_adjust_after_epoch():
    f = adjust_after_epoch(10, before=1, after=2)

    assert 1 == f(epoch=9), 'adjust_after_epoch returns wrong value before pivot'
    assert 2 == f(epoch=10), 'adjust_after_epoch returns wrong value at pivot'
    assert 2 == f(epoch=11), 'adjust_after_epoch returns wrong value after pivot'

def test_adjust_at_epochs():
    f = adjust_at_epochs([5, 10, 15], [1, 2, 3, 4])

    assert 1 == f(epoch=4), 'adjust_at_epochs returns wrong value before first pivot'
    assert 2 == f(epoch=5), 'adjust_at_epochs returns wrong value at first pivot'
    assert 2 == f(epoch=6), 'adjust_at_epochs returns wrong value after first pivot'

    assert 3 == f(epoch=10), 'adjust_at_epochs returns wrong value at internal pivot'
    assert 3 == f(epoch=11), 'adjust_at_epochs returns wrong value after internal pivot'

    assert 4 == f(epoch=15), 'adjust_at_epochs returns wrong value at last pivot'
    assert 4 == f(epoch=16), 'adjust_at_epochs returns wrong value after last pivot'

def test_invalid_adjust_at_epochs():
    exception = False

    try:
        f = adjust_at_epochs([2, 3], [4, 5])
    except Exception:
        exception = True

    assert exception, 'Invalid number of epochs and values does not throw exception'