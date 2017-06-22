import os
import pytest
import numpy as np

from tfwrapper import ImagePreprocessor

from utils import create_tmp_img


def test_imagepreprocessor_resize():
    try:
        img, path = create_tmp_img(size=(10, 10))

        import cv2
        preprocessor = ImagePreprocessor()
        preprocessor.resize_to = (64, 64)
        imgs, _ = preprocessor.process(cv2.imread(path), 'name')

        assert 1 == len(imgs), 'ImagePreprocessor does not return correct number of agumented images when performing resize'
        assert (64, 64, 3) == imgs[0].shape, 'ImagePreprocessor does not resize images'
    finally:
        os.remove(path)


def test_imagepreprocessor_bw():
    try:
        img, path = create_tmp_img(size=(10, 10))

        import cv2
        preprocessor = ImagePreprocessor()
        preprocessor.bw = True
        imgs, _ = preprocessor.process(cv2.imread(path), 'name')

        assert 1 == len(imgs), 'ImagePreprocessor does not return correct number of agumented images when performing black and white'

        img = imgs[0]
        height, width, _ = img.shape

        for i in range(height):
            for j in range(width):
                assert img[i][j][0] == img[i][j][1] == img[i][j][2], 'ImagePreprocessor does not black and white images'
        
    finally:
        os.remove(path)

def test_imagepreprocessor_flip_lr():
    try:
        img, path = create_tmp_img(size=(10, 10))

        import cv2
        preprocessor = ImagePreprocessor()
        preprocessor.flip_lr = True
        imgs, _ = preprocessor.process(cv2.imread(path), 'name')

        assert 2 == len(imgs), 'ImagePreprocessor does not return correct number of agumented images when performing flip left right'
        assert np.array_equal(np.fliplr(imgs[0]), imgs[1]), 'ImagePreprocessor does not flip horizontally correct'
    finally:
        os.remove(path)

def test_imagepreprocessor_flip_ud():
    try:
        img, path = create_tmp_img(size=(10, 10))

        import cv2
        preprocessor = ImagePreprocessor()
        preprocessor.flip_ud = True
        imgs, _ = preprocessor.process(cv2.imread(path), 'name')

        assert 2 == len(imgs), 'ImagePreprocessor does not return correct number of agumented images when performing flip up down'
        assert np.array_equal(np.flipud(imgs[0]), imgs[1]), 'ImagePreprocessor does not flip vertically correct'
    finally:
        os.remove(path)

def test_imagepreprocessor_rotate():
    try:
        steps = 2
        img, path = create_tmp_img(size=(10, 10))

        import cv2
        preprocessor = ImagePreprocessor()
        preprocessor.rotate(steps, max_rotation_angle=90)
        imgs, _ = preprocessor.process(cv2.imread(path), 'name')

        assert (2 * steps) + 1 == len(imgs), 'ImagePreprocessor does not return correct number of agumented images when performing rotate'
    finally:
        os.remove(path)

def test_imagepreprocessor_blur():
    try:
        steps = 2
        img, path = create_tmp_img(size=(10, 10))

        import cv2
        preprocessor = ImagePreprocessor()
        preprocessor.blur(steps, max_blur_sigma=5)
        imgs, _ = preprocessor.process(cv2.imread(path), 'name')

        assert steps + 1 == len(imgs), 'ImagePreprocessor does not return correct number of agumented images when performing rotate'
    finally:
        os.remove(path)