import h5py

from tfwrapper import logger
from tfwrapper import TFSession
from tfwrapper.nets import ResNet50
from tfwrapper.layers import vgg_preprocessing

from .utils import RESNET50_PATH
from .utils import download_resnet50


class PretrainedResNet50(ResNet50):
    def __init__(self, X_shape, y_size=1000, path=RESNET50_PATH, preprocessing=[], sess=None, name='PretrainedResNet50', **kwargs):
        preprocessing += vgg_preprocessing(name=name)

        with TFSession(sess) as sess:
            super().__init__(X_shape, y_size, preprocessing=preprocessing, sess=sess, name=name)

        path = download_resnet50(RESNET50_PATH)
        self.load_from_h5(path, sess=sess)

    def load_from_h5(self, path=RESNET50_PATH, sess=None):
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
                logger.debug('Loading weights and biases')
                for variable in layers:
                    name = layers[variable]
                    weight = f[name][name + '_W:0'][()]
                    bias = f[name][name + '_b:0'][()]

                    self.assign_variable_value('/'.join([self.name, variable, 'W']), weight, sess=sess)
                    self.assign_variable_value('/'.join([self.name, variable, 'b']), bias, sess=sess)
                logger.debug('Done!')

                logger.debug('Loading gammas and betas')
                for variable in batch_normalization:
                    name = batch_normalization[variable]
                    weight = f[name][name + '_beta:0'][()]
                    bias = f[name][name + '_gamma:0'][()]

                    self.assign_variable_value('/'.join([self.name, variable, 'beta']), weight, sess=sess)
                    self.assign_variable_value('/'.join([self.name, variable, 'gamma']), bias, sess=sess)
                logger.debug('Done!')

                logger.debug('Injected values into %s' % self.name)