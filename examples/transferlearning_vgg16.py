import numpy as np
import tensorflow as tf

from tfwrapper import ImageLoader
from tfwrapper import ImagePreprocessor
from tfwrapper.nets.pretrained import PretrainedVGG16
from tfwrapper.datasets import imagenet
from tfwrapper.datasets import cats_and_dogs

preprocessor = ImagePreprocessor()
preprocessor.resize_to = (224, 224)
cats_and_dogs = cats_and_dogs(size=100)
cats_and_dogs.loader = ImageLoader(preprocessor=preprocessor)

_, labels = imagenet(include_labels=True)

with tf.Session() as sess:
    vgg = PretrainedVGG16([224, 224, 3], name='PretrainedVGG16', sess=sess)

    data = cats_and_dogs.X
    prev = None

    tensor = vgg.get_tensor(-2)
    data = vgg.run_op(tensor, source=prev, data=data, sess=sess)
    print('Name: %s, shape: %s' % (tensor.name, repr(data.shape)))
    prev = tensor