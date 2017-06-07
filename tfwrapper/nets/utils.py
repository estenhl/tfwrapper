import os

from tfwrapper import config
from tfwrapper.utils.files import download_from_google_drive


VGG16_NPY_PATH = os.path.join(config.MODELS, 'vgg16.npy')
VGG16_GDRIVE_ID = '0B7gC90mSfjC6YVBUWDdZWlhaUnc'


def ensure_vgg16_npy(path=VGG16_NPY_PATH):
    if not os.path.isfile(path):
        download_from_google_drive(VGG16_GDRIVE_ID, path)

    return path


RESNET50_H5_PATH = os.path.join(config.MODELS, 'resnet50.h5')
RESNET50_DOWNLOAD_DRIVE_ID = '0B7gC90mSfjC6R19OX05vek9EN3c'


def ensure_resnet50_h5(path=RESNET50_H5_PATH):
    if not os.path.isfile(path):
        download_from_google_drive(RESNET50_DOWNLOAD_DRIVE_ID, path)

    return path