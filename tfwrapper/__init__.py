__version__ = '0.1.0-rc2'

from .logger import logger
from .config import config
from .tfsession import TFSession

# TODO (06.06.17): These are deprecated
from .dataset import Dataset
from .dataset import ImageDataset
from .dataset import ImageLoader
from .dataset import FeatureLoader
from .dataset import ImagePreprocessor

METADATA_SUFFIX = 'tw'