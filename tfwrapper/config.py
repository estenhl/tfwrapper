import os
from pathlib import Path
from tfwrapper.utils.file import file_util
CONFIG_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT = str(Path(CONFIG_PATH).parent)

DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(DATA, "models")
DATASETS = os.path.join(DATA, "datasets")

file_util.safe_mkdir(DATA)
file_util.safe_mkdir(MODELS)
file_util.safe_mkdir(DATASETS)