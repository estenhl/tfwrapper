from .dataset import Dataset
from .dataset import ImageDataset
from .dataset import ImageLoader
from .dataset import FeatureLoader
from .dataset import ImagePreprocessor
from .supervisedmodel import TFSession
from .supervisedmodel import SupervisedModel

import logging

# create logger
logger = logging.getLogger('tfwrapper')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s', '%H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
