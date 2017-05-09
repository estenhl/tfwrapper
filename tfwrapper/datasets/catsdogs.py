import os
import zipfile
import shutil

from tfwrapper import config
from tfwrapper import logger
from tfwrapper.utils.file import file_util
from tfwrapper.utils.download import google_drive

DOWNLOAD_DRIVE_ID = "0B1b2bIlebXOqQnJWYUxDZXRhTlE"
FILE_PATH = os.path.join(config.DATASETS, "catsdogs")
IMAGES = os.path.join(FILE_PATH, "images")

def download_cats_and_dogs():
    if not os.path.exists(FILE_PATH):
        logger.info('Downloading cats vs dogs dataset')

        file_util.safe_mkdir(config.DATA)
        file_util.safe_mkdir(FILE_PATH)

        tmp_dir = os.path.join(FILE_PATH, "tmp")
        file_util.safe_mkdir(tmp_dir)

        zip_destination = os.path.join(tmp_dir, "train.zip")
        google_drive.download_file_from_google_drive(DOWNLOAD_DRIVE_ID, zip_destination)

        file_util.safe_mkdir(IMAGES)
        cats = os.path.join(IMAGES, "cat")
        dogs = os.path.join(IMAGES, "dog")
        file_util.safe_mkdir(cats)
        file_util.safe_mkdir(dogs)

        with zipfile.ZipFile(zip_destination, 'r') as f:
            logger.info('Extracting dataset cats_and_dogs')
            f.extractall(tmp_dir)

        tmp_images_dir = os.path.join(tmp_dir, "train")

        logger.info("Unzipped {} files".format(len(os.listdir(tmp_images_dir))))
        for image in os.listdir(tmp_images_dir):
            if image.startswith("cat"):
                shutil.move(os.path.join(tmp_images_dir, image), os.path.join(cats, image))
            elif image.startswith("dog"):
                shutil.move(os.path.join(tmp_images_dir, image), os.path.join(dogs, image))
            else:
                logger.warning("Invalid file: {}".format(image))

        shutil.rmtree(tmp_dir)

    return IMAGES
