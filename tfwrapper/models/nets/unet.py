import tensorflow as tf

from tfwrapper.layers import unet_block
from tfwrapper.layers import zoom
from tfwrapper.layers import maxpool2d
from tfwrapper.layers import deconv2d
from tfwrapper.layers import concatenate

from .cnn import CNN


# https://arxiv.org/pdf/1505.04597.pdf
class UNet(CNN):
    def __init__(self, X_shape, classes, sess=None, name='UNet'):
        self.name = name

        wrapper = namedtuple('LayerWrapper', ['layer', 'dependencies'])
        layers = [
            wrapper(unet_block(depth=64, name=name+'/block1'), []),
            wrapper(zoom(size=(392, 392), name=name+'/block1_slice'), ['block1']),
            wrapper(maxpool2d(k=2, name=name+'/pool1'), ['block1']),

            wrapper(unet_block(depth=128, name=name+'/block2'), ['pool1']),
            wrapper(zoom(size=(200, 200), name=name+'/block2_slice'), ['block2']),
            wrapper(maxpool2d(k=2, name=name+'/pool2'), ['block2']),

            wrapper(unet_block(depth=256, name=name+'/block3'), ['pool2']),
            wrapper(zoom(size=(104, 104), name=name+'/block3_slice'), ['block3']),
            wrapper(maxpool2d(k=2, name=name+'/pool3'), ['block3']),

            wrapper(unet_block(depth=512, name=name+'/block4'), ['pool3']),
            wrapper(zoom(size=(56, 56), name=name+'/block4_slice'), ['block4']),
            wrapper(maxpool2d(k=2, name=name+'/pool4'), ['block4']),

            wrapper(unet_block(depth=1024, name=name+'/block5'), ['pool5']),
            wrapper(deconv2d(filter=(2, 2), depth=512, name=name+'/deconv1'), ['block5']),

            wrapper(concatenate(name=name+'/concat1'), ['block4_slice', 'deconv1']),
            wrapper(unet_block(depth=512, name=name+'/block6'), ['concat1']),
            wrapper(deconv2d(filter=(2, 2), depth=256, name=name+'/deconv2'), ['block6']),

            wrapper(concatenate(name=name+'/concat2'), ['block3_slice', 'deconv2']),
            wrapper(unet_block(depth=256, name=name+'/block7'), ['concat2']),
            wrapper(deconv2d(filter=(2, 2), depth=128, name=name+'/deconv3'), ['block7']),

            wrapper(concatenate(name=name+'/concat3'), ['block2_slice', 'deconv3']),
            wrapper(unet_block(depth=128, name=name+'/block8'), ['concat3']),
            wrapper(deconv2d(filter=(2, 2), depth=64, name=name+'/deconv4'), ['block8']),

            wrapper(concatenate(name=name+'/concat4'), ['block1_slice', 'deconv4']),
            wrapper(unet_block(depth=64, name=name+'/block9'), ['concat4']),

            wrapper(conv2d(filter=[1, 1], depth=classes, name=name+'/pred')}
        ]