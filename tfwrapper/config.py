import os

from pathlib import Path
from tfwrapper.utils.files import safe_mkdir

class Config():
	permit_downloads = True

	def __init__(self, root):
		if root is None:
			self.permit_downloads = False
		else:
			self.DATA = os.path.join(root, 'data')
			self.MODELS = os.path.join(self.DATA, 'models')
			self.DATASETS = os.path.join(self.DATA, 'datasets')

			safe_mkdir(root)
			safe_mkdir(self.DATA)
			safe_mkdir(self.MODELS)
			safe_mkdir(self.DATASETS)

ROOT_PATH = os.path.join(os.path.expanduser('~'), '.tfwrapper')
PERMIT_DOWNLOADS = True
config = Config(ROOT_PATH, PERMIT_DOWNLOADS)
