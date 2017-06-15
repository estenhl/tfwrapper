import pytest

from tfwrapper.utils.decorators import deprecated


class A():
    @deprecated('new_func')
    def deprecated_func(self, arg, keyword_arg=1):
        return arg * keyword_arg * 5

    def new_func(self, arg, keyword_arg=1):
        return arg * keyword_arg * 7

def new_func(arg, keyword_arg=1):
    return arg * keyword_arg * 7


@deprecated('new_func')
def deprecated1(arg, keyword_arg=1):
    return arg * keyword_arg * 5


@deprecated('invalid_func')
def deprecated2(arg, keyword_arg=1):
    return arg * keyword_arg * 11


@deprecated
def deprecated3(arg, keyword_arg=1):
    return arg * keyword_arg * 13


def test_deprecated_in_class():
    a = A()
    result = a.deprecated_func(2, keyword_arg=3)

    assert result == 2 * 3 * 7, 'A deprecated func does not call the replacement given as a decorator parameter'


def test_deprecated_outside_class():
    result = deprecated1(2, keyword_arg=3)

    assert result == 2 * 3 * 7


def test_deprecated_with_invalid_replacement():
    result = deprecated2(2, keyword_arg=3)

    assert result == 2 * 3 * 11


def test_deprecated_without_replacement():
    result = deprecated3(2, keyword_arg=3)

    assert result == 2 * 3 * 13
