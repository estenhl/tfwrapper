import numpy as np
import tensorflow as tf

from tfwrapper import ImageLoader
from tfwrapper import ImagePreprocessor
from tfwrapper.nets.pretrained import PretrainedVGG16
from tfwrapper.datasets import imagenet
from tfwrapper.datasets import cats_and_dogs

preprocessor = ImagePreprocessor()
preprocessor.resize_to = (224, 224)
cats_and_dogs = cats_and_dogs(size=5)[:10]
cats_and_dogs.loader = ImageLoader(preprocessor=preprocessor)

_, labels = imagenet(include_labels=True)

with tf.Session() as sess:
    vgg = PretrainedVGG16([224, 224, 3], sess=sess)

    cat_preds = vgg.predict(cats_and_dogs.X, sess=sess)
    print('Cat prediction: %s' % labels[np.argmax(cat_preds)])
