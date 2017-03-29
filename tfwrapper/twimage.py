
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimage


def imread(file_path):
    image = cv2.imread(file_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return rgb_image


def imwrite(file_path, image):
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, bgr_image)


def bw(image, shape=1):
    allowed_shapes = [1, 3]
    if not shape in allowed_shapes:
        raise ValueError('Illegal argument shape. Must be one of {}'.format(allowed_shapes))
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    return gray_image


def resize(image, shape=(299, 299)):
    return cv2.resize(image, shape)


def show(image):
    plt.imshow(image)
    plt.show()