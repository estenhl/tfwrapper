import os
import zipfile
import shutil

from tfwrapper import config
from tfwrapper import logger
from tfwrapper.utils.files import safe_mkdir
from tfwrapper.utils.files import download_from_google_drive
from tfwrapper.utils.exceptions import IllegalStateException

DOWNLOAD_DRIVE_ID = "0B1b2bIlebXOqQnJWYUxDZXRhTlE"
FILE_PATH = os.path.join(config.DATASETS, "catsdogs")
IMAGES = os.path.join(FILE_PATH, "images")

def parse_cats_and_dogs(size=25000):
    if size is not 25000:
        logger.warning('Size parameter is not implemented for cats and dogs dataset')
        
    if not os.path.exists(FILE_PATH):
        logger.info('Downloading cats vs dogs dataset')

        safe_mkdir(config.DATA)
        safe_mkdir(FILE_PATH)

        tmp_dir = os.path.join(FILE_PATH, "tmp")
        safe_mkdir(tmp_dir)

        zip_destination = os.path.join(tmp_dir, "train.zip")
        download_from_google_drive(DOWNLOAD_DRIVE_ID, zip_destination)

        safe_mkdir(IMAGES)
        cats = os.path.join(IMAGES, "cat")
        dogs = os.path.join(IMAGES, "dog")
        safe_mkdir(cats)
        safe_mkdir(dogs)

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
