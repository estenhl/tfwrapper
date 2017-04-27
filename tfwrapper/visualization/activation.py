import numpy as np 
import matplotlib.pyplot as plt

from tfwrapper.utils.exceptions import InvalidArgumentException

def visualize_activation(features, img, cmap='OrRd'):
    if not len(features.shape) == 3:
        raise InvalidArgumentException('Requires features on the form (height, width, depth)')

    height, width, depth = features.shape
    weights = features.sum(axis=2)
    weights = weights / np.amax(weights)

    height, width, _ = img.shape
    scaled_weights = np.zeros((height, width))
    horizontal_step = height / features.shape[0]
    vertical_step = width / features.shape[1]
    for i in range(height):
        for j in range(width):
            scaled_weights[i][j] = weights[int(i / horizontal_step)][int(j / vertical_step)]

    figure = plt.figure(figsize=(10, 10))
    ax = figure.add_subplot(111)
    plt.axis('off')
    ax.imshow(img)
    ax.imshow(scaled_weights, alpha=0.25, cmap=cmap)
    plt.show()