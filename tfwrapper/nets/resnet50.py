import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.layers import random_crop 
from tfwrapper.layers import flip_left_right
from tfwrapper.layers import hue
from tfwrapper.layers import contrast
from tfwrapper.layers import saturation
from tfwrapper.layers import normalize_image
from tfwrapper.layers import conv2d
from tfwrapper.layers import batch_normalization
from tfwrapper.layers import relu
from tfwrapper.layers import residual_block
from tfwrapper.layers import maxpool2d
from tfwrapper.layers import flatten
from tfwrapper.layers import fullyconnected
from tfwrapper.layers import softmax

from .cnn import CNN

class ResNet50(CNN):
    DEFAULT_BOTTLENECK_LAYER = -3
    
    def __init__(self, X_shape, classes, name='ResNet50', sess=None, **kwargs):
        residual_filters = [[1, 1], [3, 3], [1, 1]]
        
        depths = [
            [64, 64, 256],
            [128, 128, 512],
            [256, 256, 1024],
            [512, 512, 2048]
        ]

        seed=12345

        layers = [
            conv2d(filter=[7, 7], depth=64, strides=[2, 2], name=name + '/conv1'),
            batch_normalization(name=name + '/norm1'),
            relu(name=name + '/relu1'),
            maxpool2d(k=3, strides=2, name=name + '/maxpool'),
            
            residual_block(filters=residual_filters, depths=depths[0], activation='relu', name=name + '/residual1'),
            residual_block(filters=residual_filters, depths=depths[0], activation='relu', name=name + '/residual2'),
            residual_block(filters=residual_filters, depths=depths[0], activation='relu', name=name + '/residual3'),
            
            residual_block(filters=residual_filters, depths=depths[1], activation='relu', name=name + '/residual4'),
            residual_block(filters=residual_filters, depths=depths[1], activation='relu', name=name + '/residual5'),
            residual_block(filters=residual_filters, depths=depths[1], activation='relu', name=name + '/residual6'),
            residual_block(filters=residual_filters, depths=depths[1], activation='relu', name=name + '/residual7'),
            
            residual_block(filters=residual_filters, depths=depths[2], activation='relu', name=name + '/residual8'),
            residual_block(filters=residual_filters, depths=depths[2], activation='relu', name=name + '/residual9'),
            residual_block(filters=residual_filters, depths=depths[2], activation='relu', name=name + '/residual10'),
            residual_block(filters=residual_filters, depths=depths[2], activation='relu', name=name + '/residual11'),
            residual_block(filters=residual_filters, depths=depths[2], activation='relu', name=name + '/residual12'),
            residual_block(filters=residual_filters, depths=depths[2], activation='relu', name=name + '/residual13'),
            
            residual_block(filters=residual_filters, depths=depths[3], activation='relu', name=name + '/residual14'),
            residual_block(filters=residual_filters, depths=depths[3], activation='relu', name=name + '/residual15'),
            residual_block(filters=residual_filters, depths=depths[3], activation='relu', name=name + '/residual16'),
            
            flatten(method='avgpool', name=name + '/flatten'),
            fullyconnected(inputs=2048, outputs=classes, name=name + '/fc'),
            softmax(name=name + '/pred')
        ]

        with TFSession(sess) as sess:
            super().__init__(X_shape, classes, layers, sess=sess, name=name, **kwargs)