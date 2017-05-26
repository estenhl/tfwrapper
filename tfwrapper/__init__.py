__version__ = '0.0.7'

import os

from .logger import logger

if os.path.isfile(os.path.join(os.path.dirname(__file__), os.pardir, '.path')):
    from .config import config
else:
    logger.error('tfwrapper is not configured! Run ./configure in the root folder')
    exit()

from .dataset import Dataset
from .dataset import ImageDataset
from .dataset import ImageLoader
from .dataset import FeatureLoader
from .dataset import ImagePreprocessor
from .tfsession import TFSession
from .supervisedmodel import SupervisedModel