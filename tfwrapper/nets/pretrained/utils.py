import os
import tarfile
from shutil import copyfile

from tfwrapper.utils.files import remove_dir
from tfwrapper.utils.files import download_file

if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)),'models')):
	os.mkdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models'))

INCEPTION_PB_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','inception_v3.pb')
INCEPTION_TAR_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
INCEPTION_TAR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models', 'inception_v3.tar')
INCEPTION_PB_TEMP_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','inception-2015-12-05', 'classify_image_graph_def.pb')

def inceptionv3_pb_path(verbose=True):
	if not os.path.isfile(INCEPTION_PB_PATH):
		if not os.path.isfile(INCEPTION_PB_TEMP_PATH):
			if not os.path.isfile(INCEPTION_TAR_PATH):
				download_file(INCEPTION_TAR_URL, INCEPTION_TAR_PATH, verbose=verbose)

		with tarfile.open(INCEPTION_TAR_PATH, 'r') as f:
			if verbose:
				print('Extracting Inception V3 pb file')

			if not os.path.isdir(os.path.dirname(INCEPTION_PB_TEMP_PATH)):
				os.mkdir(os.path.dirname(INCEPTION_PB_TEMP_PATH))
			for item in f:
				f.extract(item, os.path.dirname(INCEPTION_PB_TEMP_PATH))

		copyfile(INCEPTION_PB_TEMP_PATH, INCEPTION_PB_PATH)
			
		if os.path.isdir(os.path.dirname(INCEPTION_PB_TEMP_PATH)):
			remove_dir(os.path.dirname(INCEPTION_PB_TEMP_PATH))
		if os.path.isfile(INCEPTION_TAR_PATH):
			os.remove(INCEPTION_TAR_PATH)

	return INCEPTION_PB_PATH

VGG16_CKPT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','vgg_16.ckpt')
VGG16_TAR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','vgg16.tar')
VGG16_TAR_URL = 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz'

def vgg16_ckpt_path(verbose=True):
	if not os.path.isfile(VGG16_CKPT_PATH):
		if not os.path.isfile(VGG16_TAR_PATH):
			download_file(VGG16_TAR_URL, VGG16_TAR_PATH, verbose=verbose)

		with tarfile.open(VGG16_TAR_PATH, 'r') as f:
			if verbose:
				print('Extracting VGG16 checkpoint')
			for item in f:
				f.extract(item, os.path.dirname(VGG16_CKPT_PATH))

		os.remove(VGG16_TAR_PATH)

	return VGG16_CKPT_PATH

SSD300_CKPT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','ssd_300_vgg.ckpt')
SSD300_CKPT_URL = 'https://github.com/balancap/SSD-Tensorflow/raw/master/checkpoints/ssd_300_vgg.ckpt.zip'

def ssd300_ckpt_path(verbose=True):
	if not os.path.isfile(SSD300_CKPT_PATH):
		download_file(SSD300_CKPT_URL, SSD300_CKPT_PATH, verbose=verbose)

	return SSD300_CKPT_PATH
