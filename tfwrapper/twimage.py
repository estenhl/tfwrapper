
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage

def _read_gif(filepath):
    return scipy.ndimage.imread(filepath)


def imread(filepath):
    if filepath.endswith('.gif'):
        return _read_gif(filepath)

    rgb_image = None
    
    try:
        image = cv2.imread(filepath)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(e)

    return rgb_image



def imwrite(file_path, image):
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, bgr_image)


def blur(image, sigma):
    return scipy.ndimage.filters.gaussian_filter(image, sigma)


def bw(image, shape=1):
    allowed_shapes = [1, 3]
    if not shape in allowed_shapes:
        raise ValueError('Illegal argument shape. Must be one of {}'.format(allowed_shapes))
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if shape == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    return gray_image


def resize(image, shape=(299, 299)):
    return cv2.resize(image, shape)


def rotate(image, angle):
    return scipy.ndimage.interpolation.rotate(image, angle, reshape=False)


def show(image):
    plt.imshow(image)
    plt.show()