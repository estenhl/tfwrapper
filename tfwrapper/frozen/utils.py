import os

from tfwrapper import config
from tfwrapper.utils.files import download_from_google_drive


def _download_if_necessary(path, drive_id):
    if not os.path.isfile(path):
        download_from_google_drive(drive_id, path)

    return path


VGG16_PB_PATH = os.path.join(config.MODELS, 'vgg16.pb')
VGG16_GDRIVE_ID = '0B7gC90mSfjC6djh3b3Yxd2l2M3M'


def ensure_vgg16_pb(path=VGG16_PB_PATH):
    return _download_if_necessary(path, VGG16_GDRIVE_ID)


INCEPTIONV3_PB_PATH = os.path.join(config.MODELS, 'inception_v3.pb')
INCEPTIONV3_GDRIVE_ID = '0B1b2bIlebXOqcXJTaDh6YXFfM1U'


def ensure_inception_v3_pb(path=INCEPTIONV3_PB_PATH):
    return _download_if_necessary(path, INCEPTIONV3_GDRIVE_ID)


INCEPTIONV4_PB_PATH = os.path.join(config.MODELS, 'inception_v4.pb')
INCEPTIONV4_GDRIVE_ID = '0B1b2bIlebXOqN3JWdHRZc05xdzQ'


def ensure_inception_v4_pb(path=INCEPTIONV4_PB_PATH):
    return _download_if_necessary(path, INCEPTIONV4_GDRIVE_ID)


RESNET50_PB_PATH = os.path.join(config.MODELS, 'resnet_v2_50.pb')
RESNET50_GDRIVE_ID = '0B7gC90mSfjC6SS1nbXVXZkJBbGc'


def ensure_resnet50_pb(path=RESNET50_PB_PATH):
    return _download_if_necessary(path, RESNET50_GDRIVE_ID)


RESNET152_PB_PATH = os.path.join(config.MODELS, 'resnet_v2_152.pb')
RESNET152_GDRIVE_ID = '0B7gC90mSfjC6NFJpQzNoTHhFdEk'


def ensure_resnet152_pb(path=RESNET152_PB_PATH):
    return _download_if_necessary(path, RESNET152_GDRIVE_ID)