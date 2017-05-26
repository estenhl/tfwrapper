import os

from pathlib import Path

class Config():
    permit_downloads = True

    def __init__(self, root):
        if root is None:
            self.permit_downloads = False
            root = '/tmp/tfwrapper'

        self.DATA = os.path.join(root, 'data')
        self.MODELS = os.path.join(self.DATA, 'models')
        self.DATASETS = os.path.join(self.DATA, 'datasets')

        if not os.path.isdir(root):
            os.mkdir(root)
        if not os.path.isdir(self.DATA):
            os.mkdir(self.DATA)
        if not os.path.isdir(self.MODELS):
            os.mkdir(self.MODELS)
        if not os.path.isdir(self.DATASETS):
            os.mkdir(self.DATASETS)

ROOT_PATH = None
path_file = os.path.join(os.path.dirname(__file__), os.pardir, '.path')
if os.path.isfile(path_file):
    with open(path_file, 'r') as f:
        ROOT_PATH = f.read().strip()

config = Config(ROOT_PATH)
