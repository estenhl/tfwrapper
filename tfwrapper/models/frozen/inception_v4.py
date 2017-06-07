from tfwrapper import TFSession
from tfwrapper.models import FrozenModel

from .utils import INCEPTIONV4_PB_PATH
from .utils import ensure_inception_v4_pb


class FrozenInceptionV4(FrozenModel):
    input_tensor = 'input:0'
    output_tensor = 'InceptionV4/Logits/Predictions:0'
    bottleneck_tensor = 'InceptionV4/Logits/PreLogitsFlatten/Reshape:0'

    def __init__(self, path=INCEPTIONV4_PB_PATH, name='FrozenInceptionV4', sess=None):
        path = ensure_inception_v4_pb(path)
        
        with TFSession(sess) as sess:
            super().__init__(path, name=name, sess=sess)