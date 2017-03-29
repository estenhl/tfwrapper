import tensorflow as tf

from tfwrapper.datasets import cats_and_dogs
from tfwrapper.nets.pretrained import PretrainedVGG16

X_shape = [224, 224, 3]
with tf.Session() as sess:
	cnn = PretrainedVGG16(X_shape, sess=sess, graph=sess.graph)