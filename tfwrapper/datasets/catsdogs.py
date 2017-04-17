import os
import zipfile
import cv2
import shutil

from tfwrapper.utils.file import file_util
from tfwrapper.utils.download import google_drive
from tfwrapper import config
from tfwrapper.containers import ImageContainer


DOWNLOAD_DRIVE_ID = "0B1b2bIlebXOqQnJWYUxDZXRhTlE"
FILE_PATH = os.path.join(config.DATA, "catsdogs")
IMAGES = os.path.join(FILE_PATH, "images")

def download_cats_and_dogs(verbose=False):

    if os.path.exists(FILE_PATH):
        print("Already downloaded catsvsdogs dataset")
        return

    print("Downloading catsvsdogs dataset")

    file_util.safe_mkdir(config.DATA)
    file_util.safe_mkdir(FILE_PATH)

    tmp_dir = os.path.join(FILE_PATH, "tmp")
    file_util.safe_mkdir(tmp_dir)

    zip_destionation = os.path.join(tmp_dir, "train.zip")
    google_drive.download_file_from_google_drive(DOWNLOAD_DRIVE_ID, zip_destionation)

    print("Completed downloading dataset")

    file_util.safe_mkdir(IMAGES)
    cats = os.path.join(IMAGES, "cat")
    dogs = os.path.join(IMAGES, "dog")
    file_util.safe_mkdir(cats)
    file_util.safe_mkdir(dogs)

    with zipfile.ZipFile(zip_destionation, 'r') as f:
        print('Extracting dataset cats_and_dogs')
        f.extractall(tmp_dir)

    tmp_images_dir = os.path.join(tmp_dir, "train")

    print("Unzipped {} files".format(len(os.listdir(tmp_images_dir))))
    for image in os.listdir(tmp_images_dir):
        if image.startswith("cat"):
            shutil.move(os.path.join(tmp_images_dir, image), os.path.join(cats, image))
        elif image.startswith("dog"):
            shutil.move(os.path.join(tmp_images_dir, image), os.path.join(dogs, image))
        else:
            print("Invalid file: {}".format(image))

    print("Cleaning up")
    shutil.rmtree(tmp_dir)

    print("Complete")

def create_container(max_images=10000):
    container = ImageContainer(dir_path=IMAGES)

    class_count = max_images / 2
    container = container.balance(max_value=class_count)

    return container