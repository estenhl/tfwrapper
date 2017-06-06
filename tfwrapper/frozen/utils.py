import os

from tfwrapper import config
from tfwrapper.utils.files import download_from_google_drive

VGG16_PB_PATH = os.path.join(config.DATASETS, 'vgg16.pb')
VGG16_GDRIVE_ID = '0B7gC90mSfjC6djh3b3Yxd2l2M3M'

def download_vgg16_pb(path=VGG16_PB_PATH):
    if not os.path.isfile(path):
        download_from_google_drive(VGG16_GDRIVE_ID, path)

    return path