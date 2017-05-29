import os
from .logger import logger


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

if 'TFWRAPPER_ROOT_PATH' in os.environ:
    print(os.environ['TFWRAPPER_ROOT_PATH'])
    ROOT_PATH = os.environ['TFWRAPPER_ROOT_PATH']
else:
    ROOT_PATH = os.path.join(os.path.expanduser('~'), '.tfwrapper')
    logger.warning('TFWRAPPER_ROOT_PATH not set. Using: ' + str(ROOT_PATH))

logger.info('Testing logging')
config = Config(ROOT_PATH)
