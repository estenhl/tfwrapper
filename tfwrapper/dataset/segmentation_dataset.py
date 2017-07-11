import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tfwrapper import logger
from tfwrapper import twimage
from tfwrapper.utils.exceptions import raise_exception

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
            height, width, channels = img.shape

            if img is not None:
                images.append(img)
        except Exception:
            logger.warning('Skipping %s' % filename)

    return np.asarray(images)


def onehot_encode_image(img):
    height, width = img.shape
    num_classes = int(np.amax(img)) + 1
    encoded = np.zeros((height, width, num_classes))

    for i in range(height):
        for j in range(width):
            label = int(img[i][j])
            encoded[i][j][label] = 1

    return encoded


def _get_pixel_value(encoded):
    r = (encoded & 0x00ff0000) >> 16
    g = (encoded & 0x0000ff00) >> 8
    b = encoded & 0x000000ff

    return np.asarray([r, g, b])


def translate_image_labels(imgs):
    translated_imgs = imgs.copy()
    translated_imgs = translated_imgs.astype(np.int32)

    translated_imgs[:,:,:,0] = np.left_shift(translated_imgs[:,:,:,0], 16)
    translated_imgs[:,:,:,1] = np.left_shift(translated_imgs[:,:,:,1], 8)

    translated_imgs = np.sum(translated_imgs, axis=3)

    unique = np.unique(translated_imgs)
    labels = []
    for i in range(len(unique)):
        pixel_value = _get_pixel_value(unique[i])
        labels.append(pixel_value)
        translated_imgs[translated_imgs==unique[i]] = i

    return translated_imgs, labels

class SegmentationDataset(Dataset):
    """ 
    A class for representing datasets used by segmentation models.

    Typical usage:
        dataset = SegmentationDataset.from_root_folder(root_folder) # A folder containing an imgs and a labels folder with matching filenames
        dataset = dataset.resized(max_size=(MODEL.OUTPUT_HEIGHT, MODEL.OUTPUT_WIDTH))
        dataset = dataset.squarepadded()
        dataset = dataset.framed_X((MODEL.INPUT_HEIGHT-MODEL.OUTPUT_HEIGHT, MODEL.INPUT_WIDTH-MODEL.OUTPUT_HEIGHT))
        dataset = dataset.translated_labels()
        dataset = dataset.onehot_encoded()
        MODEL.TRAIN(dataset.X, dataset.y)

        Args:
            X (np.ndarray): The images of the dataset
            y (np.ndarray): The labels of the dataset
            background_labels (list): The pixel values that should be treated as background. (Defaults to [np.asarray([0., 0., 0.])])

    """
    def __init__(self, X, y, **kwargs):
        super().__init__(X=X, y=y, **kwargs)

    @classmethod
    def from_root_folder(cls, root_folder, img_folder_name='imgs', labels_folder_name='labels', size=None):
        X = parse_images(os.path.join(root_folder, img_folder_name), max_size=size)
        y = parse_images(os.path.join(root_folder, labels_folder_name), max_size=size)

        return SegmentationDataset(X=X, y=y)

    def normalized(self, columnwise=False):
        raise_exception('Unable to perform straight normalization of a Segmentation Dataset (Imagewise normalization not implemented)', NotImplementedError)

    def onehot_encoded(self):
        y = []
        for img in self._y:
            y = onehot_encode_image(img)

        return self.__class__(X=self._X, y=np.asarray(y), labels=self.labels)

    def translated_labels(self):
        """ Translates the (typically 3-channeled) pixel values of a dataset
        into single class labels """

        y, labels = translate_image_labels(self._y)

        return self.__class__(X=self._X, y=y, labels=labels)

    def visualize(self, num=None):
        """ Visualizes the first |num| images of the dataset, with the labels overlayed """
        
        if num is None:
            num = len(self._X)

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
            X.append(cv2.copyMakeBorder(self._X[i], top, bottom, left, right, method))

        return self.__class__(X=np.asarray(X), y=self._y)

