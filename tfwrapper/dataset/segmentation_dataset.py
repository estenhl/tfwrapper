import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tfwrapper import logger
from tfwrapper import twimage

from .dataset import Dataset


def parse_images(folder, max_size=None):
    logger.info('Parsing images from %s' % folder)
    images = []

    i = 0
    for filename in os.listdir(folder):
        i += 1
        if max_size is not None and i > max_size:
            break
        try:
            img = twimage.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
        except Exception:
            logger.warning('Skipping %s' % filename)

    return np.asarray(images)


class SegmentationDataset(Dataset):
    """ 
    A class for representing datasets used by segmentation models.

    Typical usage:
        dataset = SegmentationDataset.from_root_folder(root_folder) # A folder containing an imgs and a labels folder with matching filenames
        dataset = dataset.resized(max_size=(MODEL.OUTPUT_HEIGHT, MODEL.OUTPUT_WIDTH))
        dataset = dataset.squarepadded()
        dataset = dataset.framed_X((MODEL.INPUT_HEIGHT-MODEL.OUTPUT_HEIGHT, MODEL.INPUT_WIDTH-MODEL.OUTPUT_HEIGHT))
        MODEL.TRAIN(dataset.X, dataset.y)
    """
    def __init__(self, X, y):
        super().__init__(X=X, y=y)

    @classmethod
    def from_root_folder(cls, root_folder, img_folder_name='imgs', labels_folder_name='labels', size=None):
        X = parse_images(os.path.join(root_folder, img_folder_name), max_size=size)
        y = parse_images(os.path.join(root_folder, labels_folder_name), max_size=size)

        return SegmentationDataset(X=X, y=y)

    def visualize(self, num):
        """ Visualizes the first |num| images of the dataset, with the labels overlayed """

        for i in range(min(num, len(self._X))):
            figure = plt.figure()
            plt.imshow(self._X[i])
            X_height, X_width, _ = self._X[i].shape
            y_height, y_width, _ = self._y[i].shape

            # Shows the label as an overlay if X and y are equally shaped
            if (X_height, X_width) == (y_height, y_width):
                plt.imshow(self._y[i], alpha=0.5)

            plt.show()

    def resized(self, *, max_size):
        """ Resizes both the images and the labels of the dataset. Nearest neighbors is used
        when resizing the labels to maintain categorical class labels """

        X = []
        y = []
        new_height, new_width = max_size

        for i in range(len(self._X)):
            height, width, _ = self._X[i].shape
            height_ratio = height / new_height
            width_ratio = width / new_width
            ratio = max(height_ratio, width_ratio)
            new_size = (int(width / ratio), int(height / ratio))
            X.append(cv2.resize(self._X[i], new_size))
            y.append(cv2.resize(self._y[i], new_size, interpolation=cv2.INTER_NEAREST))

        return self.__class__(X=np.asarray(X), y=np.asarray(y))

    def squarepadded(self, method=cv2.BORDER_REFLECT):
        """ Pads each image and label as needed to produce a pair of squares. Labels 
        are padded in such a way that any object of the image which is reflected in the
        generated portion of the image has its corresponding label (Atleast when using the 
        default method) """

        X = []
        y = []

        for i in range(len(self._X)):
            height, width, _ = self._X[i].shape
            size = abs(height - width)

            # Computes an array representing number of pixels to pad to [top, bottom, left, right]
            axis = np.argmax([height, width])
            axis = np.asarray([0, 0, 1, 1]) - axis
            axis = np.abs(axis)
            axis = axis * (size / 2)

            # Increases one element (top/left) and decreases one element(bottom/right) whenever the 
            # number of pixels is not an even number
            if size % 2 != 0:
                axis = np.rint(axis + [0.1, -0.1, 0.1, -0.1])

            axis = axis.astype(int)
            
            top, bottom, left, right = axis
            X.append(cv2.copyMakeBorder(self._X[i], top, bottom, left, right, method))
            y.append(cv2.copyMakeBorder(self._y[i], top, bottom, left, right, method))

        return self.__class__(X=np.asarray(X), y=np.asarray(y))

    def framed_X(self, size, method=cv2.BORDER_REFLECT):
        """ Frames the images with a padding. Labels are NOT padded equivalently, so 
        one will often end up with labels and images of different sizes """

        X = []
        height, width = size
        top = int(height / 2)
        bottom = int(height / 2)
        left = int(width / 2)
        right = int(width / 2)

        if height % 2 != 0:
            top += 1

        if width % 2 != 0:
            left += 1

        for i in range(len(self._X)):
            print('Before: ' + str(self._X[i].shape))
            X.append(cv2.copyMakeBorder(self._X[i], top, bottom, left, right, method))
            print('After: ' + str(X[i].shape))

        return self.__class__(X=np.asarray(X), y=self._y)

