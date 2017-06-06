from tfwrapper import TFSession

from .frozenmodel import FrozenModel
from .utils import VGG16_PB_PATH
from .utils import download_vgg16_pb

class FrozenVGG16(FrozenModel):
    def __init__(self, path=VGG16_PB_PATH, name='FrozenVGG16', sess=None):
        path = download_vgg16_pb(path)

        input_tensor = 'vgg_16/Input:0'
        output_tensor = 'vgg_16/Prediction:0'
        bottleneck_tensor = 'vgg_16/fc7/Relu:0'
        
        with TFSession(sess) as sess:
            super().__init__(path, input_tensor=input_tensor, output_tensor=output_tensor, bottleneck_tensor=bottleneck_tensor, name=name, sess=sess)