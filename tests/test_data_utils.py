from tfwrapper.utils.data import get_subclass_by_name
from tfwrapper.utils.exceptions import InvalidArgumentException

class A():
    val = 1

class B(A):
    val = 2

def test_get_subclass():
    subclass = get_subclass_by_name(A, 'B')

    assert 2 == subclass.val

def test_get_invalid_subclass():
    exception = False

    try:
        subclass = get_subclass_by_name(A, 'C')
    except InvalidArgumentException:
        exception = True

    assert exception

