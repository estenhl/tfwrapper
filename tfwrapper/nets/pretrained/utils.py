import os

from tfwrapper.utils.files import download_file

SSD300_CKPT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','ssd_300_vgg.ckpt')
SSD300_CKPT_URL = 'https://github.com/balancap/SSD-Tensorflow/raw/master/checkpoints/ssd_300_vgg.ckpt.zip'
INCEPTION_PB_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models','inception_v3.pb')

def inceptionv3_pb_path():
	if not os.path.isfile(INCEPTION_PB_PATH):
		raise NotImplementedError('Download of inception is not implemented')

	return INCEPTION_PB_PATH

def ssd300_ckpt_path(verbose=False):
	if not os.path.isfile(SSD300_CKPT_PATH):
		download_file(SSD300_CKPT_URL, SSD300_CKPT_PATH, verbose=verbose)

	return SSD300_CKPT_PATH
