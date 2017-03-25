import tensorflow as tf

from tfwrapper import ImageTransformer
from tfwrapper.nets import SSD300_VGG
from tfwrapper.datasets import mnist

dataset = mnist(size=1000, verbose=True)
transformer = ImageTransformer(resize_to=(300, 300), rgb=True)
X, y, test_X, test_y, _, _ = dataset.getdata(normalize=True, split=True, transformer=transformer)

with tf.Session() as sess:
	cnn = SSD300_VGG(sess=sess, graph=sess.graph, name='ExampleSSD300_VGG')
	cnn.train(X, y, epochs=5, sess=sess)