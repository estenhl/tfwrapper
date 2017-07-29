import os
import shutil
import tarfile

from tfwrapper import config
from tfwrapper import logger
from tfwrapper.utils.files import remove_dir
from tfwrapper.utils.files import download_file
from tfwrapper.utils.exceptions import raise_exception


def _prompt_user():
    response = input('Advancing requires a download of VOC2012 (>2gb). Proceed? (Y/n): ')
    
    if response.lower() in ['', 'y', 'yes']:
        return True

    return False


def _copy_data(root_folder, name, file_dir):
    data_folder = os.path.join(root_folder, 'data')
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)

    folder = os.path.join(root_folder, 'data', name)
    if not os.path.isdir(folder):
        os.mkdir(folder)

    dest_img_folder = os.path.join(folder, 'imgs')
    if not os.path.isdir(dest_img_folder):
        os.mkdir(dest_img_folder)

    dest_labels_folder = os.path.join(folder, 'labels')
    if not os.path.isdir(dest_labels_folder):
        os.mkdir(dest_labels_folder)

    src_img_folder = os.path.join(root_folder, 'VOCdevkit', 'VOC2012', 'JPEGImages')
    src_labels_folder = os.path.join(root_folder, 'VOCdevkit', 'VOC2012', 'SegmentationClass')

    datafile = os.path.join(file_dir, '%s.txt' % name)
    with open(datafile, 'r') as f:
        for line in f.readlines():
            name = line.strip()

            dest_img_file = os.path.join(dest_img_folder, '%s.jpg' % name)
            src_img_file = os.path.join(src_img_folder, '%s.jpg' % name)
            shutil.copyfile(src_img_file, dest_img_file)

            dest_labels_file = os.path.join(dest_labels_folder, '%s.png' % name)
            src_labels_file = os.path.join(src_labels_folder, '%s.png' % name)
            shutil.copyfile(src_labels_file, dest_labels_file)


def parse_voc2012(subset='train'):
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    prompted = False

    root_folder = os.path.join(config.DATASETS, 'VOC2012')
    if not os.path.isdir(root_folder):
        if not _prompt_user():
            raise_exception('Unable to proceed without downloading VOC2012 dataset', ValueError)
        prompted = True
        
        os.mkdir(root_folder)

    data_folder = os.path.join(root_folder, 'data')
    if not os.path.isdir(data_folder):
        if (not prompted) and not _prompt_user():
            raise_exception('Unable to proceed without downloading VOC2012 dataset', ValueError)
        prompted = True

        tar_file = os.path.join(root_folder, 'VOCtrainval_11-May-2012.tar')
        if not os.path.isfile(tar_file):
            download_file(url, tar_file)

        os.mkdir(data_folder)

        with tarfile.open(tar_file, 'r') as f:
            for item in f:
                f.extract(item, root_folder)

        file_dir = os.path.join(root_folder, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation')
        for name in ['train', 'trainval', 'val']:
            _copy_data(root_folder, name, file_dir)

        os.remove(tarfile)
        remove_dir('VOCdevkit')

    return os.path.join(data_folder, subset)
