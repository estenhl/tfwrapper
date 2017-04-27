import os
import functools

from tfwrapper import ImageDataset

def kfold_imagesize_validation(dataset, image_sizes, k=10):
    accuracys = []
    training_durations = []
    prediction_durations = []
    model_sizes = []

    folds = dataset.folds(k)
    for image_size in image_sizes:
        accuracy = []
        training_duration = []
        prediction_duration = []
        model_size = []
        for i in range(k):
            testset = folds[i]
            trainset = functools.reduce(lambda x, y: x + y, folds[:i] + folds[i + 1:],)
            print('Testlength: ' + str(len(testset)))
            print('Trainlength: ' + str(len(trainset)))
            print('Totallength: ' + str(len(dataset)))

if __name__ == '__main__':
    curr_path = os.path.abspath(os.path.join(os.path.realpath(__file__), '..'))
    data_path = os.path.join(curr_path, '..', '..', 'data', 'datasets', 'catsdogs', 'images')

    dataset = ImageDataset(root_folder=data_path)[:100]
    kfold_imagesize_validation(dataset, [(166, 166), (88, 88)])
