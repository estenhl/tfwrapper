import numpy as np
import tensorflow as tf

from tfwrapper import ImageLoader
from tfwrapper import ImagePreprocessor
from tfwrapper.models.nets import ResNet50
from tfwrapper.datasets import imagenet
from tfwrapper.datasets import cats_and_dogs

preprocessor = ImagePreprocessor()
preprocessor.resize_to = (224, 224)
cats_and_dogs = cats_and_dogs(size=5)[:10]
cats_and_dogs.loader = ImageLoader(preprocessor=preprocessor)

_, labels = imagenet(include_labels=True)

with tf.Session() as sess:
    resnet = ResNet50.from_h5(X_shape=[224, 224, 3], sess=sess)

    cat_preds = resnet.predict(cats_and_dogs.X, sess=sess)
    for i in range(len(cat_preds)):
        preds = [(j, cat_preds[i][j]) for j in range(len(cat_preds[i]))]
        preds = sorted(preds, key=lambda x: x[1], reverse=True)
        print([labels[j[0]] for j in preds[:5]])
