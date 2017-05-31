import h5py

from tfwrapper import logger
from tfwrapper import TFSession
from tfwrapper.nets import ResNet50

from .utils import RESNET50_PATH
from .utils import download_resnet50


class PretrainedResNet50(ResNet50):
    def __init__(self, X_shape, path=RESNET50_PATH, sess=None, name='PretrainedResNet50'):
        with TFSession(sess) as sess:
            super().__init__(X_shape, 1000, sess=sess, name=name)

        path = download_resnet50(RESNET50_PATH)
        self.load_from_h5(path, sess=sess)

    def load_from_h5(self, path=RESNET50_PATH, sess=None):
        with h5py.File(path, 'r') as f:
            logger.debug('Loaded ResNet50 from %s' % path)
            variables = {
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
                'pred': 'fc1000'
            }
            with TFSession(sess, self.graph, init=True) as sess:
                for variable in variables:
                    name = variables[variable]
                    weight = f[name][name + '_W:0'][()]
                    bias = f[name][name + '_b:0'][()]

                    self.assign_variable_value('/'.join([self.name, variable, 'W']), weight, sess=sess)
                    self.assign_variable_value('/'.join([self.name, variable, 'b']), bias, sess=sess)

                logger.debug('Injected values into %s' % self.name)