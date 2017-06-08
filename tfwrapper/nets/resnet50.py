import tensorflow as tf

from tfwrapper import TFSession
from tfwrapper.layers import vgg_preprocessing
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
from .utils import RESNET50_H5_PATH
from .utils import ensure_resnet50_h5

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
            super().from_shape(X_shape, classes, layers, sess=sess, name=name, **kwargs)

    def load_from_h5(self, path, sess=None):
        with h5py.File(path, 'r') as f:
            logger.debug('Loaded ResNet50 from %s' % path)

            layers = {
                'conv1': 'conv1',
                'residual1/shortcut': 'res2a_branch1',
                'residual1/module_0/conv': 'res2a_branch2a',
                'residual1/module_1/conv': 'res2a_branch2b',
                'residual1/module_2/conv': 'res2a_branch2c',
                'residual2/module_0/conv': 'res2b_branch2a',
                'residual2/module_1/conv': 'res2b_branch2b',
                'residual2/module_2/conv': 'res2b_branch2c',
                'residual3/module_0/conv': 'res2c_branch2a',
                'residual3/module_1/conv': 'res2c_branch2b',
                'residual3/module_2/conv': 'res2c_branch2c',
                'residual4/shortcut': 'res3a_branch1',
                'residual4/module_0/conv': 'res3a_branch2a',
                'residual4/module_1/conv': 'res3a_branch2b',
                'residual4/module_2/conv': 'res3a_branch2c',
                'residual5/module_0/conv': 'res3b_branch2a',
                'residual5/module_1/conv': 'res3b_branch2b',
                'residual5/module_2/conv': 'res3b_branch2c',
                'residual6/module_0/conv': 'res3c_branch2a',
                'residual6/module_1/conv': 'res3c_branch2b',
                'residual6/module_2/conv': 'res3c_branch2c',
                'residual7/module_0/conv': 'res3d_branch2a',
                'residual7/module_1/conv': 'res3d_branch2b',
                'residual7/module_2/conv': 'res3d_branch2c',
                'residual8/shortcut': 'res4a_branch1',
                'residual8/module_0/conv': 'res4a_branch2a',
                'residual8/module_1/conv': 'res4a_branch2b',
                'residual8/module_2/conv': 'res4a_branch2c',
                'residual9/module_0/conv': 'res4b_branch2a',
                'residual9/module_1/conv': 'res4b_branch2b',
                'residual9/module_2/conv': 'res4b_branch2c',
                'residual10/module_0/conv': 'res4c_branch2a',
                'residual10/module_1/conv': 'res4c_branch2b',
                'residual10/module_2/conv': 'res4c_branch2c',
                'residual11/module_0/conv': 'res4d_branch2a',
                'residual11/module_1/conv': 'res4d_branch2b',
                'residual11/module_2/conv': 'res4d_branch2c',
                'residual12/module_0/conv': 'res4e_branch2a',
                'residual12/module_1/conv': 'res4e_branch2b',
                'residual12/module_2/conv': 'res4e_branch2c',
                'residual13/module_0/conv': 'res4f_branch2a',
                'residual13/module_1/conv': 'res4f_branch2b',
                'residual13/module_2/conv': 'res4f_branch2c',
                'residual14/shortcut': 'res5a_branch1',
                'residual14/module_0/conv': 'res5a_branch2a',
                'residual14/module_1/conv': 'res5a_branch2b',
                'residual14/module_2/conv': 'res5a_branch2c',
                'residual15/module_0/conv': 'res5b_branch2a',
                'residual15/module_1/conv': 'res5b_branch2b',
                'residual15/module_2/conv': 'res5b_branch2c',
                'residual16/module_0/conv': 'res5c_branch2a',
                'residual16/module_1/conv': 'res5c_branch2b',
                'residual16/module_2/conv': 'res5c_branch2c',
                'fc': 'fc1000'
            }

            batch_normalization = {
                'norm1': 'bn_conv1',
                'residual1/shortcut/norm': 'bn2a_branch1',
                'residual1/module_0/norm': 'bn2a_branch2a',
                'residual1/module_1/norm': 'bn2a_branch2b',
                'residual1/module_2/norm': 'bn2a_branch2c',
                'residual2/module_0/norm': 'bn2b_branch2a',
                'residual2/module_1/norm': 'bn2b_branch2b',
                'residual2/module_2/norm': 'bn2b_branch2c',
                'residual3/module_0/norm': 'bn2c_branch2a',
                'residual3/module_1/norm': 'bn2c_branch2b',
                'residual3/module_2/norm': 'bn2c_branch2c',
                'residual4/shortcut/norm': 'bn3a_branch1',
                'residual4/module_0/norm': 'bn3a_branch2a',
                'residual4/module_1/norm': 'bn3a_branch2b',
                'residual4/module_2/norm': 'bn3a_branch2c',
                'residual5/module_0/norm': 'bn3b_branch2a',
                'residual5/module_1/norm': 'bn3b_branch2b',
                'residual5/module_2/norm': 'bn3b_branch2c',
                'residual6/module_0/norm': 'bn3c_branch2a',
                'residual6/module_1/norm': 'bn3c_branch2b',
                'residual6/module_2/norm': 'bn3c_branch2c',
                'residual7/module_0/norm': 'bn3d_branch2a',
                'residual7/module_1/norm': 'bn3d_branch2b',
                'residual7/module_2/norm': 'bn3d_branch2c',
                'residual8/shortcut/norm': 'bn4a_branch1',
                'residual8/module_0/norm': 'bn4a_branch2a',
                'residual8/module_1/norm': 'bn4a_branch2b',
                'residual8/module_2/norm': 'bn4a_branch2c',
                'residual9/module_0/norm': 'bn4b_branch2a',
                'residual9/module_1/norm': 'bn4b_branch2b',
                'residual9/module_2/norm': 'bn4b_branch2c',
                'residual10/module_0/norm': 'bn4c_branch2a',
                'residual10/module_1/norm': 'bn4c_branch2b',
                'residual10/module_2/norm': 'bn4c_branch2c',
                'residual11/module_0/norm': 'bn4d_branch2a',
                'residual11/module_1/norm': 'bn4d_branch2b',
                'residual11/module_2/norm': 'bn4d_branch2c',
                'residual12/module_0/norm': 'bn4e_branch2a',
                'residual12/module_1/norm': 'bn4e_branch2b',
                'residual12/module_2/norm': 'bn4e_branch2c',
                'residual13/module_0/norm': 'bn4f_branch2a',
                'residual13/module_1/norm': 'bn4f_branch2b',
                'residual13/module_2/norm': 'bn4f_branch2c',
                'residual14/shortcut/norm': 'bn5a_branch1',
                'residual14/module_0/norm': 'bn5a_branch2a',
                'residual14/module_1/norm': 'bn5a_branch2b',
                'residual14/module_2/norm': 'bn5a_branch2c',
                'residual15/module_0/norm': 'bn5b_branch2a',
                'residual15/module_1/norm': 'bn5b_branch2b',
                'residual15/module_2/norm': 'bn5b_branch2c',
                'residual16/module_0/norm': 'bn5c_branch2a',
                'residual16/module_1/norm': 'bn5c_branch2b',
                'residual16/module_2/norm': 'bn5c_branch2c',
            }

            # If the model does not have 1000 ouputs, pretrained weights for the final layer are dropped
            if not self.y_size == 1000:
                del layers['fc']

            with TFSession(sess, self.graph, init=True) as sess:
                logger.debug('Injecting weights and biases into %s' % self.name)
                for variable in layers:
                    name = layers[variable]
                    weight = f[name][name + '_W:0'][()]
                    bias = f[name][name + '_b:0'][()]

                    self.assign_variable_value('/'.join([self.name, variable, 'W']), weight, sess=sess)
                    self.assign_variable_value('/'.join([self.name, variable, 'b']), bias, sess=sess)
                logger.debug('Done!')

                logger.debug('Injecting gammas and betas into %s' % self.name)
                for variable in batch_normalization:
                    name = batch_normalization[variable]
                    weight = f[name][name + '_beta:0'][()]
                    bias = f[name][name + '_gamma:0'][()]

                    self.assign_variable_value('/'.join([self.name, variable, 'beta']), weight, sess=sess)
                    self.assign_variable_value('/'.join([self.name, variable, 'gamma']), bias, sess=sess)
                logger.debug('Done!')

    @staticmethod
    def from_h5(path=RESNET50_H5_PATH, *, X_shape, classes=1000, name='PretrainedResNet50', sess=None):
        path = ensure_resnet50_h5(path)
        with TFSession(sess) as sess:
            model = ResNet50(X_shape, classes, preprocessing=vgg_preprocessing(), name=name, sess=sess)
            model.load_from_h5(path, sess)