from tfwrapper.layers import channel_means, conv2d, maxpool2d, fullyconnected, relu, dropout, softmax

from .cnn import CNN
from .utils import VGG16_NPY_PATH
from .utils import ensure_vgg16_npy

class VGG16(CNN):
    DEFAULT_BOTTLENECK_LAYER = -5

    def __init__(self, X_shape, classes=1000, sess=None, name='VGG16'):
        height, width, channels = X_shape
        
        if not ((height % 2 ** 5) == 0 and (width % 2 ** 5) == 0):
            raise ValueError('Height and width must be divisible by %d' % 2 ** 5)
        
        fc_input_size = int(height / (2**5)) * int(width / (2**5)) * 512

        layers = [
            conv2d(filter=[3, 3], depth=64, name=name + '/conv1_1'),
            conv2d(filter=[3, 3], depth=64, name=name + '/conv1_2'),
            maxpool2d(k=2, name=name + '/pool1'),
            conv2d(filter=[3, 3], depth=128, name=name + '/conv2_1'),
            conv2d(filter=[3, 3], depth=128, name=name + '/conv2_2'),
            maxpool2d(k=2, name=name + '/pool2'),
            conv2d(filter=[3, 3], depth=256, name=name + '/conv3_1'),
            conv2d(filter=[3, 3], depth=256, name=name + '/conv3_2'),
            conv2d(filter=[3, 3], depth=256, name=name + '/conv3_3'),
            maxpool2d(k=2, name=name + '/pool3'),
            conv2d(filter=[3, 3], depth=512, name=name + '/conv4_1'),
            conv2d(filter=[3, 3], depth=512, name=name + '/conv4_2'),
            conv2d(filter=[3, 3], depth=512, name=name + '/conv4_3'),
            maxpool2d(k=2, name=name + '/pool4'),
            conv2d(filter=[3, 3], depth=512, name=name + '/conv5_1'),
            conv2d(filter=[3, 3], depth=512, name=name + '/conv5_2'),
            conv2d(filter=[3, 3], depth=512, name=name + '/conv5_3'),
            maxpool2d(k=2, name=name + '/pool5'),
            fullyconnected(inputs=fc_input_size, outputs=4096, name=name + '/fc6'),
            relu(name=name + '/relu1'),
            # TODO (06.06.17: Should be replaced with a real keep_prob value
            dropout(keep_prob=1.0, name=name + '/dropout1'),
            fullyconnected(inputs=4096, outputs=4096, name=name + '/fc7'),
            relu(name=name + '/relu2'),
            # TODO (06.06.17: Should be replaced with a real keep_prob value
            dropout(keep_prob=1.0, name=name + '/dropout2'),
            fullyconnected(inputs=4096, outputs=classes, name=name + '/fc8'),
            softmax(name=name + '/pred')
        ]

        super().from_shape(X_shape, classes, layers, sess=sess, name=name)

    def load_from_npy(self, path, sess=None):
        with TFSession(sess, self.graph) as sess:
            data = np.load(path, encoding='latin1').item()
            logger.debug('Loaded VGG16 from %s' % npy_path)
            for key in data:
                root_name = '/'.join([self.name, key])
                weight_name = '/'.join([root_name, 'W'])
                bias_name = '/'.join([root_name, 'b'])

                self.assign_variable_value(weight_name, data[key][0], sess=sess)
                self.assign_variable_value(bias_name, data[key][1], sess=sess)

            logger.debug('Injected values into %s' % self.name)

    @staticmethod
    def from_npy(path=VGG16_NPY_PATH, name='PretrainedVGG16', sess=None):
        path = ensure_vgg16_npy(path)
        with TFSession(sess) as sess:
            model = VGG16([224, 224, 3], 1000, name=name, sess=sess)
            model.load_from_npy(path, sess=sess)