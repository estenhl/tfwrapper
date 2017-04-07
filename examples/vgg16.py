import cv2
import numpy as np
import tensorflow as tf

from tfwrapper.nets import VGG16
from tfwrapper.datasets import mnist


X_shape = [224, 224, 3]
data = mnist(size=1000, verbose=True)
X, y, test_X, test_y, _, _ = data.getdata(balance=True, split=True, onehot=True, translate_labels=True)
X_reshaped = np.zeros((len(X), 224, 224, 3))
for i in range(len(X)):
	X_reshaped[i] = cv2.cvtColor(cv2.resize(X[i], (224, 224)), cv2.COLOR_GRAY2BGR)
X = X_reshaped
with tf.Session() as sess:
	cnn = VGG16(X_shape, classes=10, sess=sess)
	cnn.train(X, y, epochs=10, sess=sess)