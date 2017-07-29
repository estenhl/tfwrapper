import os
import pytest

from tfwrapper import twimage

from utils import curr_path


def test_read_jpg():
    img_path = os.path.join(curr_path, 'data', 'dog.jpg')
    img = twimage.imread(img_path)

    assert img is not None

def test_read_gif():
    img_path = os.path.join(curr_path, 'data', 'cat.gif')
    img = twimage.imread(img_path)

    assert img is not None