import os
import time
import functools
import numpy as np
import tensorflow as tf

from tfwrapper import ImageLoader
from tfwrapper import ImageDataset
from tfwrapper import ImagePreprocessor
from tfwrapper.nets import ShallowCNN

def kfold_imagesize_validation(dataset, image_sizes, k=10):
    accuracies = []
    training_durations = []
    prediction_durations = []
    model_sizes = []

    for image_size in image_sizes:
        accuracy = []
        training_duration = []
        prediction_duration = []
        model_size = []

        dataset = dataset.translate_labels()

        preprocessor = ImagePreprocessor()
        preprocessor.resize_to = image_size
        dataset.loader = ImageLoader(preprocessor)
        folds = dataset.folds(k)

        for i in range(k):
            test = folds[i]
            train = functools.reduce(lambda x, y: x + y, folds[:i] + folds[i + 1:],)
            train = train.onehot()
            test = test.onehot()

            with tf.Session() as sess:
                cnn = ShallowCNN([image_size[0], image_size[1], 3], 2, name='Val%d' % i, sess=sess)
                cnn.learning_rate = 0.0001

                train_start = time.time()
                cnn.train(train.X, train.y, epochs=10, verbose=True, sess=sess)
                training_duration.append(time.time() - train_start)

                val_start = time.time()
                accuracy.append(cnn.validate(test.X, test.y, sess=sess)[1])
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
        accuracies.append(np.mean(np.asarray(accuracy)))
        training_durations.append(np.mean(np.asarray(training_duration)))
        prediction_durations.append(np.mean(np.asarray(prediction_duration)))
        #model_sizes.append(np.mean(np.asarray(model_size)))

    return accuracies, training_durations, prediction_durations, model_sizes
if __name__ == '__main__':
    curr_path = os.path.abspath(os.path.join(os.path.realpath(__file__), '..'))
    data_path = os.path.join(curr_path, '..', '..', 'data', 'datasets', 'catsdogs', 'images')

    dataset = ImageDataset(root_folder=data_path)
    dataset = dataset.shuffle()
    #dataset = dataset[:100]
    print(kfold_imagesize_validation(dataset, [(100, 100), (4, 4)], k=5))
