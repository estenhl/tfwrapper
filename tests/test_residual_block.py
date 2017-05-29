import pytest

from tfwrapper.layers import residual_block

def test_block_with_filters():
    exception = False
    try:
        residual_block(modules=3, filters=[[1, 1], [2, 2], [3, 3]], depths=16)
    except Exception as e:
        print(e)
        exception = True
    assert not exception

def test_block_with_invalid_filters():
    exception = False
    try:
        residual_block(modules=2, filters=[[1, 1], [2, 2], [3, 3]], depths=16)
    except Exception:
        exception = True
    assert exception


def test_block_with_single_filter():
    exception = False
    try:
        residual_block(filters=[2, 3], depths=16)
    except Exception as e:
        print(e)
        exception = True
    assert not exception


def test_block_with_strides():
    exception = False
    try:
        residual_block(modules=3, filters=[3, 3], strides=[[1, 1], [2, 2], [3, 3]], depths=16)
    except Exception:
        exception = True
    assert not exception


def test_block_invalid_strides():
    exception = False
    try:
        residual_block(modules=2, filters=[3, 3], strides=[[1, 1], [2, 2], [3, 3]], depths=16)
    except Exception:
        exception = True
    assert exception


def test_block_with_single_strides():
    exception = False
    try:
        residual_block(filters=[2, 3], strides=[1, 1], depths=16)
    except Exception as e:
        print(e)
        exception = True
    assert not exception

def test_block_with_depths():
    exception = False
    try:
        residual_block(modules=3, filters=[3, 3], depths=[16, 16, 16])
    except Exception as e:
        print(e)
        exception = True
    assert not exception


def test_block_invalid_depths():
    exception = False
    try:
        residual_block(modules=2, filters=[3, 3], depths=[16, 16, 16])
    except Exception:
        exception = True
    assert True
