import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from time import process_time

from tfwrapper.nets import RNN
from tfwrapper.datasets import mnist

dataset = mnist(size=1000)
dataset = dataset.normalize()
dataset = dataset.translate_labels()
dataset = dataset.onehot()
# Set the random seed so that dataset.shuffle becomes deterministic, hence allowing us to split into the same sets
# each time and thus allowing us to compare across reruns of this script
np.random.seed(4)  # RFC 1149.5
dataset = dataset.shuffle()
train_ds, test_ds = dataset.split()

learning_rate = 0.001
batch_size = 128
epochs = 25
nr_of_runs = 10

# Per training run statistics.
time = []   # time per training run
acc = []    # test accuracy after each run
lloss = []  # logloss each run
conf = []   # confusion matrix for each run

# Todo: This as a function, and a second model to compare to would be nice
# Train the model several times to get a better idea of its performance
for rep in range(nr_of_runs):  # statistics gathering loop
    tf.reset_default_graph()
    model = RNN([28], 28, 128, 10, name='RNN_example')
    model.learning_rate = learning_rate
    model.batch_size = batch_size

    time_start = process_time()
    model.train(train_ds.X, train_ds.y, epochs=epochs, verbose=True, validate=True)
    time_end = process_time()
    time.append(time_end - time_start)

    predictions = model.predict(test_ds.X)
    test_size = len(test_ds.X)

    pred_label = np.argmax(predictions, axis=1)
    real_label = np.argmax(test_ds.y, axis=1)

    accuracy = np.sum(pred_label == real_label) / test_size
    acc.append(accuracy)

    lloss.append(log_loss(test_ds.y, predictions))
    conf.append(confusion_matrix(real_label, pred_label))
if len(acc) > 0:  # todo: split into second function
    print("Raw time values:", time)
    print("Raw acc values:", acc)
    print("Raw logloss values:", lloss)

    print("\nMean time", np.mean(time), "variance", np.var(time))
    print("Mean acc", np.mean(acc), "variance", np.var(acc))
    print("Median acc", np.median(acc))
    print("Mean logloss", np.mean(lloss), "variance", np.var(lloss))

    print("Labels:", test_ds.labels)
    print("Mean confusion matrix_:")
    mean_conf = np.mean(conf, 0)
    # todo: dynamic width, then put in tfwrapper
    print("      ", end='')
    for i in range(len(test_ds.labels)):
        print("%5s " % test_ds.labels[i], end='')
    print()
    for i in range(len(test_ds.labels)):
        print("%5s " % test_ds.labels[i], end='')
        for j in range(len(mean_conf[i])):
            print("%5.1f " % mean_conf[i][j], end='')
        print()

# Example comparison of models using boxplots and wilcoxon ranksum. If multiple models are available.
# In this case acc_model1 is the raw test acccuracy list from above, while acc_model2 is the same with a
# different model / hyperparameters.
'''print(ranksums(acc_model1, acc_model2)) # if p<0.05 there is a statistically significant difference in their distributions
plt.figure()
plt.boxplot([acc_model1, acc_model2])
plt.xticks([1,2], ['model1', 'model2'])
plt.ylabel('Test accuracy')
plt.show()'''
