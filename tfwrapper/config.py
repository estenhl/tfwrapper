import os
from pathlib import Path
from tfwrapper.utils.files import safe_mkdir

CONFIG_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT = str(Path(CONFIG_PATH).parent)

DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(DATA, "models")
DATASETS = os.path.join(DATA, "datasets")

safe_mkdir(DATA)
safe_mkdir(MODELS)
safe_mkdir(DATASETS)