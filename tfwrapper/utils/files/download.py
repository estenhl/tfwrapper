import urllib.request

from tfwrapper import logger

def download_file(url, path, verbose=False):
	logger.info('Downloading ' + url  + ' to ' + path)

	urllib.request.urlretrieve(url, path)