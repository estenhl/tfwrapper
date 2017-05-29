__version__ = '0.0.8'

#import os

from .logger import logger
from .config import config

# if os.path.isfile(os.path.join(os.path.dirname(__file__), os.pardir, '.path')):
# else:
#     logger.error('tfwrapper is not configured! Run ./configure in the root folder')
#     exit()

from .dataset import Dataset
from .dataset import ImageDataset
from .dataset import ImageLoader
from .dataset import FeatureLoader
from .dataset import ImagePreprocessor
from .tfsession import TFSession
from .supervisedmodel import SupervisedModel