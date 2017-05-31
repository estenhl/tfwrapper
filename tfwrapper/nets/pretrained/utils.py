import os
import tarfile
from shutil import copyfile

from tfwrapper import config
from tfwrapper import logger
from tfwrapper.utils.files import remove_dir
from tfwrapper.utils.files import download_file
from tfwrapper.utils.files import download_from_google_drive


INCEPTION_TAR_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
INCEPTION_PB_PATH = os.path.join(config.MODELS, 'inception_v3.pb')
INCEPTION_TAR_PATH = os.path.join(config.MODELS, 'inception_v3.tar')
INCEPTION_PB_TEMP_PATH = os.path.join(config.MODELS, 'inception-2015-12-05', 'classify_image_graph_def.pb')


def download_inceptionv3(path=INCEPTION_PB_PATH):
	if not os.path.isfile(path):
		if not os.path.isfile(INCEPTION_PB_TEMP_PATH):
			if not os.path.isfile(INCEPTION_TAR_PATH):
				download_file(INCEPTION_TAR_URL, INCEPTION_TAR_PATH)

		with tarfile.open(INCEPTION_TAR_PATH, 'r') as f:
			logger.info('Extracting Inception V3 pb file')

			if not os.path.isdir(os.path.dirname(INCEPTION_PB_TEMP_PATH)):
				os.mkdir(os.path.dirname(INCEPTION_PB_TEMP_PATH))
			for item in f:
				f.extract(item, os.path.dirname(INCEPTION_PB_TEMP_PATH))

		copyfile(INCEPTION_PB_TEMP_PATH, path)
			
		if os.path.isdir(os.path.dirname(INCEPTION_PB_TEMP_PATH)):
			remove_dir(os.path.dirname(INCEPTION_PB_TEMP_PATH))
		if os.path.isfile(INCEPTION_TAR_PATH):
			os.remove(INCEPTION_TAR_PATH)

	return path


VGG16_TAR_URL = 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz'
VGG16_CKPT_PATH = os.path.join(config.MODELS, 'vgg_16.ckpt')
VGG16_TAR_PATH = os.path.join(config.MODELS, 'vgg16.tar')


def download_vgg16_ckpt(path=VGG16_CKPT_PATH):
	if not os.path.isfile(path):
		if not os.path.isfile(VGG16_TAR_PATH):
			download_file(VGG16_TAR_URL, VGG16_TAR_PATH)

		with tarfile.open(VGG16_TAR_PATH, 'r') as f:
			logger.info('Extracting VGG16 checkpoint')
			for item in f:
				f.extract(item, os.path.dirname(path))

		os.remove(VGG16_TAR_PATH)

	return path


VGG16_NPY_PATH = os.path.join(config.MODELS, 'vgg16.npy')
VGG16_DOWNLOAD_DRIVE_ID = '0B7gC90mSfjC6YVBUWDdZWlhaUnc'


def download_vgg16_npy(path=VGG16_NPY_PATH):
	if not os.path.isfile(path):
		download_from_google_drive(VGG16_DOWNLOAD_DRIVE_ID, path)

	return path


RESNET50_PATH = os.path.join(config.MODELS, 'resnet50.h5')
RESNET50_DOWNLOAD_DRIVE_ID = '0B7gC90mSfjC6R19OX05vek9EN3c'


def download_resnet50(path=RESNET50_PATH):
	if not os.path.isfile(path):
		download_from_google_drive(RESNET50_DOWNLOAD_DRIVE_ID, path)

	return path


SSD300_CKPT_URL = 'https://github.com/balancap/SSD-Tensorflow/raw/master/checkpoints/ssd_300_vgg.ckpt.zip'

