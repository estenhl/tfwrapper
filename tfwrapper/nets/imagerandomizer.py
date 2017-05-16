from tfwrapper import logger
from tfwrapper.layers import 

from .cnn import CNN

class ImageRandomizer():
    def __init__(self, seed=None, flip_ud=False, flip_lr=False, brightness_delta=None, hue_delta=None, contrast_range=None, saturation_range=None, name='ImageRandomizer'):
        layers = []

        if flip_ud:
            layers.append(flip_ud_down(seed=seed, name=name + '/random_flip_ud'))

        if flip_lr:
            layers.append(flip_left_right(seed=seed, name=name + '/random_flip_lr'))

        if brightness_delta is not None:
            layers.append(brightness(brightness_delta, seed=seed, name=name + '/random_brightness'))

        if hue_delta is not None:
            layers.append(hue(hue_delta, seed=seed, name=name + '/random_hue'))

        if contrast_range is not None:
            if len(contrast_range) == 2 and type(contrast_range[0]) == type(contrast_range[1]) == float:
                layers.append(contrast(contrast_range[0], contrast_range[1], seed=seed, name=name + '/random_contrast'))
            else:
                errormsg = 'contrast_range must be [float, float]'
                logger.error(errormsg)
                raise InvalidArgumentException(errormsg)

        if saturation_range is not None:
            if len(saturation_range) and type(saturation_range[0]) == type(saturation_range[1]) == float:
                layers.append(saturation(saturation_range[0], saturation_range[1], seed=seed, name=name + '/random_saturation'))
            else:
                errormsg = 'saturation_range must be [float, float]'
                logger.error(errormsg)
                raise InvalidArgumentException(errormsg)