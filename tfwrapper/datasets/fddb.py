import os
import tarfile
import shutil

from tfwrapper import config, logger
from tfwrapper.utils.files import remove_dir, download_file

image_url = 'http://tamaraberg.com/faceDataset/originalPics.tar.gz'
labels_url = 'http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz'

def parse_fddb():
    root_folder = os.path.join(config.DATASETS, 'FDDB')

    imgs_folder = os.path.join(root_folder, 'imgs')
    labels_folder = os.path.join(root_folder, 'labels')

    if not os.path.isdir(root_folder):
        os.mkdir(root_folder)
        os.mkdir(imgs_folder)
        os.mkdir(labels_folder)

        tmp_folder = os.path.join(root_folder, 'tmp')
        os.mkdir(tmp_folder)

        tmp_labels = os.path.join(root_folder, 'tmp', 'labels')
        os.mkdir(tmp_labels)

        images_tar = os.path.join(tmp_folder, 'originalPics.tar.gz')
        download_file(image_url, images_tar)

        labels_tar = os.path.join(tmp_folder, 'FDDB-folds.tgz')

        download_file(labels_url, labels_tar)

        logger.info('Unpacking FDDB data')
        with tarfile.open(images_tar) as f:
            for item in f:
                try:
                    f.extract(item, imgs_folder)
                except Exception:
                    pass

        with tarfile.open(labels_tar) as f:
            for item in f:
                f.extract(item, tmp_labels)

        for filename in os.listdir(os.path.join(tmp_labels, 'FDDB-folds')):
            if filename.endswith('ellipseList.txt'):
                src = os.path.join(os.path.join(tmp_labels, 'FDDB-folds', filename))
                dest = os.path.join(labels_folder, filename)
                shutil.copyfile(src, dest)
        logger.info('Finished unpacking FDDB data')

        remove_dir(tmp_folder)

    return imgs_folder, labels_folder
