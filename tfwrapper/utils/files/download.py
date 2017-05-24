import urllib.request

from tfwrapper import config
from tfwrapper import logger
from tfwrapper.utils.exceptions import IllegalStateException

def download_file(url, path, verbose=False):
    if not config.permit_downloads:
        errormsg = 'tfwrapper does not have permission to download files. Change this by running ./configure'
        logger.error(errormsg)
        raiseIllegalStateException(errormsg)

	logger.info('Downloading ' + url  + ' to ' + path)

	urllib.request.urlretrieve(url, path)