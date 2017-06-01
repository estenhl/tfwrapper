import numpy as np
import tensorflow as tf

from tfwrapper import ImageLoader
from tfwrapper import ImagePreprocessor
from tfwrapper.nets.pretrained import PretrainedResNet50
from tfwrapper.datasets import imagenet
from tfwrapper.datasets import cats_and_dogs

preprocessor = ImagePreprocessor()
preprocessor.resize_to = (224, 224)
cats_and_dogs = cats_and_dogs(size=5)[:10]
cats_and_dogs.loader = ImageLoader(preprocessor=preprocessor)

_, labels = imagenet(include_labels=True)

with tf.Session() as sess:
    resnet = PretrainedResNet50([224, 224, 3], sess=sess)

    cat_preds = resnet.predict(cats_and_dogs.X, sess=sess)
    for i in range(len(cat_preds)):
        preds = cat_preds[i].argsort()[-3:][::-1]
        print('%s: %s' % (cats_and_dogs.y[i], str(['%s, %.3f' % (labels[x], cat_preds[i][x]]) for x in preds)))
