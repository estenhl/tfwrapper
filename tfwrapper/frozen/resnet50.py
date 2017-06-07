from tfwrapper import TFSession

from .frozenmodel import FrozenModel
from .utils import RESNET50_PB_PATH
from .utils import ensure_resnet50_pb

class FrozenResNet50(FrozenModel):
    input_tensor = 'resnet_v2_50/Input:0'
    output_tensor = 'resnet_v2_50/Prediction:0'
    bottleneck_tensor = 'resnet_v2_50/pool5:0'

    def __init__(self, path=RESNET50_PB_PATH, name='FrozenResnet50', sess=None):
        path = ensure_resnet50_pb(path)
        
        with TFSession(sess) as sess:
            super().__init__(path, name=name, sess=sess)