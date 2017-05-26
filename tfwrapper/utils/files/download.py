import urllib.request

from tfwrapper import config
from tfwrapper import logger
from tfwrapper.utils.exceptions import IllegalStateException

def download_file(url, path, verbose=False):
    logger.info('Downloading %s to %s' % (url, path))

    urllib.request.urlretrieve(url, path)