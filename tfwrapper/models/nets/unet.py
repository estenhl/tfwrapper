import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.layers import Layer
from tfwrapper.layers import unet_block
from tfwrapper.layers import zoom
from tfwrapper.layers import maxpool2d
from tfwrapper.layers import deconv2d
from tfwrapper.layers import concatenate
from tfwrapper.layers import conv2d

from .cnn import CNN


# https://arxiv.org/pdf/1505.04597.pdf
class UNet(CNN):
    def __init__(self, classes, sess=None, name='UNet'):
        self.name = name

        layers = [
            unet_block(depth=64, name=name+'/block1'),
            zoom(size=(392, 392), name=name+'/block1_slice'),
            Layer(maxpool2d(k=2, name=name+'/pool1'), dependencies='block1'),

            unet_block(depth=128, name=name+'/block2'),
            zoom(size=(200, 200), name=name+'/block2_slice'),
            Layer(maxpool2d(k=2, name=name+'/pool2'), dependencies='block2'),

            unet_block(depth=256, name=name+'/block3'),
            zoom(size=(104, 104), name=name+'/block3_slice'),
            Layer(maxpool2d(k=2, name=name+'/pool3'), dependencies='block3'),

            unet_block(depth=512, name=name+'/block4'),
            zoom(size=(56, 56), name=name+'/block4_slice'),
            Layer(maxpool2d(k=2, name=name+'/pool4'), dependencies='block4'),

            unet_block(depth=1024, name=name+'/block5'),
            deconv2d(filter=[2, 2], depth=512, name=name+'/deconv1'),

            Layer(concatenate(name=name+'/concat1'), dependencies=['block4_slice', 'deconv1']),
            unet_block(depth=512, name=name+'/block6'),
            deconv2d(filter=[2, 2], depth=256, name=name+'/deconv2'),

            Layer(concatenate(name=name+'/concat2'), dependencies=['block3_slice', 'deconv2']),
            unet_block(depth=256, name=name+'/block7'),
            deconv2d(filter=[2, 2], depth=128, name=name+'/deconv3'),

            Layer(concatenate(name=name+'/concat3'), dependencies=['block2_slice', 'deconv3']),
            unet_block(depth=128, name=name+'/block8'),
            deconv2d(filter=[2, 2], depth=64, name=name+'/deconv4'),

            Layer(concatenate(name=name+'/concat4'), dependencies=['block1_slice', 'deconv4']),
            unet_block(depth=64, name=name+'/block9'),

            conv2d(filter=[1, 1], depth=classes, name=name+'/pred')
        ]

        with TFSession(sess) as sess:
            super().__init__([572, 572, 3], classes, layers, sess=sess, name=name)