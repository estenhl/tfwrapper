from tfwrapper import TFSession
from tfwrapper.models import FrozenModel

from .utils import RESNET152_PB_PATH
from .utils import ensure_resnet152_pb


class FrozenResNet152(FrozenModel):
    input_tensor = 'resnet_v2_152/Input:0'
    output_tensor = 'resnet_v2_152/Prediction:0'
    bottleneck_tensor = 'resnet_v2_152/pool5:0'

    def __init__(self, path=RESNET152_PB_PATH, name='FrozenResnet50', sess=None):
        path = ensure_resnet152_pb(path)
        
        with TFSession(sess) as sess:
            super().__init__(path, name=name, sess=sess)
