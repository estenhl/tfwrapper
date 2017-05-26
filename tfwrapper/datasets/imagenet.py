import os

from tfwrapper import config
from tfwrapper.utils.files import download_file


def parse_imagenet_label(s):
    return s.split('\'')[1]


def parse_imagenet_labels():
    url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/596b27d23537e5a1b5751d2b0481ef172f58b539/imagenet1000_clsid_to_human.txt'
    path = os.path.join(config.DATA, 'imagenet_labels.txt')

    if not os.path.isfile(path):
        download_file(url, path)

    labels = []
    with open(path, 'r') as f:
        for line in f.readlines():
            labels.append(parse_imagenet_label(line))

    return labels
