import os
import time
import functools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Validator():
    def __init__(self, names=None):
        self.names = names

        self.metrics = {}
        self.metrics['Accuracy'] = []
        self.metrics['Training duration'] = []
        self.metrics['Prediction duration'] = []
        self.metrics['Model size'] = []

    def plot(self, figsize=(10, 10), filepath=None):
        fig, ax = plt.subplots(2, 2)

        ax[0][0].set_xlim([0, len(self.names) - 1])
        ax[0][0].set_ylim([0, 1])
        ax[0][0].plot(self.metrics['Accuracy'])
        ax[0][0].set_title('Accuracy')

        """
        size_ax.set_xlim([0, len(image_sizes)-1])
        size_ax.set_ylim([0, np.amax(np.asarray(sizes))*1.1])
        size_ax.plot(sizes)
        size_ax.set_title('Model size')
        """

        ax[0][1].set_xlim([0, len(self.names) - 1])
        ax[0][1].set_ylim([0, np.amax(np.asarray(self.metrics['Training duration']))*1.1])
        ax[0][1].plot(self.metrics['Training duration'])
        ax[0][1].set_title('Training duration')

        ax[1][1].set_xlim([0, len(self.names) - 1])
        ax[1][1].set_ylim([0, np.amax(np.asarray(self.metrics['Prediction duration']))*1.1])
        ax[1][1].plot(self.metrics['Prediction duration'])
        ax[1][1].set_title('Prediction duration')

        for i in range(2):
            for j in range(2):
                plt.sca(ax[i, j])
                plt.xticks(range(len(self.names)), [str(x) for x in self.names], rotation='vertical')

        plt.tight_layout()

        if filepath is not None:
            plt.savefig(filepath)

        plt.show()

    def __str__(self):
        s = 'names = %s' % str(self.names) + '\r\n'
        s += 'accuracies = %s' % str(self.metrics['Accuracy']) + '\r\n'
        s += 'training_durations = %s' % str(self.metrics['Training duration']) + '\r\n'
        s += 'prediction_duration = %s' % str(self.metrics['Prediction duration']) + '\r\n'

        return s

def kfold_validation(dataset, create_model, k=10, epochs=10, validator=Validator()):
    dataset = dataset.translate_labels()
    folds = dataset.folds(k)

    accuracy = []
    training_duration = []
    prediction_duration = []
    model_size = []

    for i in range(k):
        test = folds[i]
        train = functools.reduce(lambda x, y: x + y, folds[:i] + folds[i + 1:],)
        train = train.onehot()
        test = test.onehot()

        with tf.Session() as sess:
            model = create_model(i, sess)
            train_start = time.time()
            model.train(train.X, train.y, epochs=epochs, sess=sess)
            training_duration.append(time.time() - train_start)

            val_start = time.time()
            accuracy.append(model.validate(test.X, test.y, sess=sess)[1])
            prediction_duration.append(time.time() - val_start)
            """
            cnn.save('test_%d' % i, sess=sess)
            model_size.append(os.stat('test_%d.data-00000-of-00001' % i).st_size)
            os.remove('checkpoint')
            os.remove('test_%d.meta' % i)
            os.remove('test_%d.index' % i)
            os.remove('test_%d.tw' % i)
            """
        tf.reset_default_graph()

    validator.metrics['Accuracy'].append(np.mean(np.asarray(accuracy)))
    validator.metrics['Training duration'].append(np.mean(np.asarray(training_duration)))
    validator.metrics['Prediction duration'].append(np.mean(np.asarray(prediction_duration)))

    return validator
