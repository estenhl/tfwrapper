import numpy as np
import tensorflow as tf

from tfwrapper import ImageLoader
from tfwrapper import ImagePreprocessor
from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.nets.pretrained import PretrainedVGG16
from tfwrapper.datasets import imagenet
from tfwrapper.datasets import cats_and_dogs

dataset = cats_and_dogs(size=100)
dataset = dataset.shuffle()
dataset = dataset[:100]
dataset = dataset.translate_labels()
dataset = dataset.onehot()
train, test = dataset.split(0.8)

preprocessor = ImagePreprocessor()
preprocessor.resize_to = (224, 224)
train.loader = ImageLoader(preprocessor=preprocessor)
test.loader = ImageLoader(preprocessor=preprocessor)

with tf.Session() as sess:
    vgg = PretrainedVGG16([224, 224, 3], name='PretrainedVGG16', sess=sess)
    vgg.batch_size = 8

    for layer in [13, 17, -8, -5]:
        tensor = vgg.get_tensor(layer)
        data = vgg.run_op(tensor, data=train.X, sess=sess)
        test_data = vgg.run_op(tensor, data=test.X, sess=sess)

        shape = data.shape
        neurons = np.prod(data.shape[1:])
        print('Neurons in layer %d: %d' % (layer, neurons))
        X = np.reshape(data, (-1, neurons))
        test_X = np.reshape(test_data, (-1, neurons))
        
        nn = SingleLayerNeuralNet([neurons], 2, 1024, sess=sess, name='VGG16Test')
        nn.train(X, train.y, epochs=10, sess=sess)
        _, acc = nn.validate(test_X, test.y, sess=sess)
        print('Acc at layer %d: %d%%' % (layer, acc * 100))

