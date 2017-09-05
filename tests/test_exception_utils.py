from tfwrapper.utils.exceptions import raise_exception
from tfwrapper.utils.exceptions import IllegalStateException
from tfwrapper.utils.exceptions import InvalidArgumentException

def test_raise_exception():
    exception = False
    try:
        raise_exception('msg', NameError)
    except NameError as e:
        exception = True
        msg = str(e)

    assert exception, 'Exception from raise_exception is not the correct type'
    assert 'msg' == msg, 'Exception from raise_exception does not contain correct error msg'


def test_illegalstateexception():
    e = IllegalStateException('msg')
    assert 'msg' == str(e), 'IllegalStateException does not contain correct error msg'

def test_invalidargumentexception():
    e = InvalidArgumentException('msg')
    assert 'msg' == str(e), 'InvalidArgumentException does not contain correct error msg'
