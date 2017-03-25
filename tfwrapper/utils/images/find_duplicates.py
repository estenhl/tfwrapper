import os
import cv2
import sys
import numpy as np

def find_duplicates(folder, verbose=False):
	images = []
	for filename in os.listdir(folder):
		if not (filename.endswith('.jpg') or filename.endswith('.png')):
			continue

		images.append({'filename': filename, 'img': cv2.imread(os.path.join(folder, filename))})
	images = sorted(images, key=lambda x: np.sum(x['img']))

	duplicates = []
	for i in range(0, len(images) - 1):
		image_sum = np.sum(images[i]['img'])
		j = i + 1
		while j < len(images) and image_sum == np.sum(images[j]['img']):
			if np.array_equal(images[i]['img'], images[j]['img']):
				duplicates.append((images[i]['filename'], images[j]['filename']))

			j += 1

	return duplicates	

if __name__ == '__main__':
	print(find_image_duplicates(sys.argv[1]))