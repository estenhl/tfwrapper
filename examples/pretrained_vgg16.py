import tensorflow as tf

from tfwrapper.nets.pretrained import PretrainedVGG16

with tf.Session() as sess:
	cnn = PretrainedVGG16([224, 224, 3], sess=sess, graph=sess.graph, name='ExamplePretrainedVGG16')