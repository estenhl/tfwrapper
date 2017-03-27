import os
import tarfile

from tfwrapper.utils.files import download_file

INCEPTION_PB_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','inception_v3.pb')

VGG16_CKPT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','vgg_16.ckpt')
VGG16_TAR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','vgg16.tar')
VGG16_TAR_URL = 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz'

SSD300_CKPT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','ssd_300_vgg.ckpt')
SSD300_CKPT_URL = 'https://github.com/balancap/SSD-Tensorflow/raw/master/checkpoints/ssd_300_vgg.ckpt.zip'

def inceptionv3_pb_path():
	if not os.path.isfile(INCEPTION_PB_PATH):
		raise NotImplementedError('Download of inception is not implemented')

	return INCEPTION_PB_PATH

def vgg16_ckpt_path(verbose=False):
	if not os.path.isfile(VGG16_CKPT_PATH):
		if not os.path.isfile(VGG16_TAR_PATH):
			download_file(VGG16_TAR_URL, VGG16_TAR_PATH, verbose=verbose)

		with tarfile.open(VGG16_TAR_PATH, 'r') as f:
			if verbose:
				print('Extracting VGG16 checkpoint')
			for item in f:
				f.extract(item, os.path.dirname(VGG16_CKPT_PATH))

		if os.path.isfile(VGG16_TAR_PATH):
			os.remove(VGG16_TAR_PATH)

	return VGG16_CKPT_PATH

def ssd300_ckpt_path(verbose=False):
	if not os.path.isfile(SSD300_CKPT_PATH):
		download_file(SSD300_CKPT_URL, SSD300_CKPT_PATH, verbose=verbose)

	return SSD300_CKPT_PATH
