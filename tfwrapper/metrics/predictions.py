import numpy as np


def loss(y, yhat):
    loss = 0

    for i in range(len(y)):
        for j in range(len(y[i])):
            loss += abs(y[i][j] - yhat[i][j])

    return loss / np.prod(y.shape)


def accuracy(y, yhat):
    correct = 0

    for i in range(len(y)):
        if np.argmax(y[i]) == np.argmax(yhat[i]):
            correct += 1

    return correct / len(y)


def confusion_matrix(y, yhat):
    matrix = np.zeros((y.shape[1], y.shape[1]))

    for i in range(len(y)):
        matrix[np.argmax(y[i])][np.argmax(yhat[i])] = matrix[np.argmax(y[i])][np.argmax(yhat[i])] + 1

    return matrix


# TODO move this to an analysis module :)
def confidence_info(y, yhat, labels):

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    hit_confidence = [[]]*len(y)
    miss_confidence = [[]]*len(y)

    hit_confidence_distribution = [0] * 21
    miss_confidence_distribution = [0] * 21

    print_list = [[label] for label in labels]

    print('Test set size: ' + str(len(y)) + '\n')

    for i in range(len(y)):
        label_class = np.argmax(y[i])
        hat_class = np.argmax(yhat[i])
        confidence = np.max(softmax(yhat[i]))

        if label_class == hat_class:
            hit_confidence[label_class].append(confidence)
            hit_confidence_distribution[int(confidence * 20)] += 1
        else:
            miss_confidence[hat_class].append(confidence)
            miss_confidence_distribution[int(confidence * 20)] += 1

    for i in range(len(print_list)):
        print_list[i].append('HIT:')
        print_list[i].append(sum(hit_confidence[i]) / len(hit_confidence[i]))
        print_list[i].append(max(hit_confidence[i]))
        print_list[i].append(min(hit_confidence[i]))
        print_list[i].append('MISS:')
        print_list[i].append(sum(miss_confidence[i]) / len(miss_confidence[i]))
        print_list[i].append(max(miss_confidence[i]))
        print_list[i].append(min(miss_confidence[i]))

    for i in range(len(print_list)):
        print(print_list[i])

    print('Miss confidence distribution')
    print(miss_confidence_distribution)

    print('Hit confidence distribution')
    print(hit_confidence_distribution)
