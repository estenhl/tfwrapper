import os
import cv2
import numpy as np

def copy_image_folder(src_folder, dest_folder, bw=False, h_flip=False, v_flip=False):
	for filename in os.listdir(src_folder):
		if filename.endswith('.jpg'):
			src = os.path.join(src_folder, filename)
			copy_image(src, dest_folder, bw=bw, h_flip=h_flip, v_flip=v_flip)

def copy_image(src_file, dest_folder, bw=False, h_flip=False, v_flip=False):
	filename = os.path.basename(src_file)
	name, suffix = filename.split('.')

	img = cv2.imread(src_file)

	if bw:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	dest = os.path.join(dest_folder, name + '.' + suffix)
	cv2.imwrite(dest, img)

	if h_flip:
		dest = os.path.join(dest_folder, name + '_hflip.' + suffix)
		cv2.imwrite(dest, np.fliplr(img))
	if v_flip:
		dest = os.path.join(dest_folder, name + '_vflip.' + suffix)
		cv2.imwrite(dest, np.flipud(img))
	if h_flip and v_flip:
		dest = os.path.join(dest_folder, name + '_hvflip.' + suffix)
		cv2.imwrite(dest, np.flipud(np.fliplr(img)))