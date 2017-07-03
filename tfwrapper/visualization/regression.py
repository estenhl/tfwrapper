import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(y, preds, plot_correct=True, **kwargs):
    if 'figsize' in kwargs:
        fig = plt.figure(figsize=kwargs['figsize'])
    else:
        fig = plt.figure()

    if 'colour' in kwargs:
        plt.scatter(y, preds, c=kwargs['colour'])
    elif 'color' in kwargs:
        plt.scatter(y, preds, c=kwargs['color'])
    else:
        plt.scatter(y, preds)

    if plot_correct:
        plt.plot([np.amin(y), np.amax(y)], [np.amin(y), np.amax(y)], 'k--')

    if 'title' in kwargs:
        fig.suptitle(kwargs['title'])

    plt.show()