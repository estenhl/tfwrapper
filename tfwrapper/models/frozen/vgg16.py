from tfwrapper import TFSession
from tfwrapper.models import FrozenModel

from .utils import VGG16_PB_PATH
from .utils import ensure_vgg16_pb


class FrozenVGG16(FrozenModel):
    input_tensor = 'vgg_16/Input:0'
    output_tensor = 'vgg_16/Prediction:0'
    bottleneck_tensor = 'vgg_16/fc7/Relu:0'

    def __init__(self, path=VGG16_PB_PATH, name='FrozenVGG16', sess=None):
        path = ensure_vgg16_pb(path)
        
        with TFSession(sess) as sess:
            super().__init__(path, name=name, sess=sess)