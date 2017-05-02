import urllib.request

def download_file(url, path, verbose=False):
	if verbose:
		print('Downloading ' + url  + ' to ' + path)

	urllib.request.urlretrieve(url, path)