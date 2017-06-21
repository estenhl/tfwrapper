import pytest

from tfwrapper import ImagePreprocessor

def test_parallell_preprocessors():
    prep1 = ImagePreprocessor()
    prep1.bw = True
    prep2 = ImagePreprocessor()

    assert prep1.bw == True
    assert prep2.bw == False