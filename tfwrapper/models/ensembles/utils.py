import numpy as np

def accumulate_predictions(preds):
    return np.sum(preds, axis=0)

def vote_predictions(preds):
    votes = np.zeros(preds.shape[1:])
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            votes[j][np.argmax(preds[i][j])] += 1

    return votes