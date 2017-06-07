__version__ = '0.0.8'

from .logger import logger
from .config import config
from .tfsession import TFSession

# TODO (06.06.17): These are deprecated
from .dataset import Dataset
from .dataset import ImageDataset
from .dataset import ImageLoader
from .dataset import FeatureLoader
from .dataset import ImagePreprocessor