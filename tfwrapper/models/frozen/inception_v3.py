from tfwrapper import TFSession
from tfwrapper.models import FrozenModel

from .utils import INCEPTIONV3_PB_PATH
from .utils import ensure_inception_v3_pb


class FrozenInceptionV3(FrozenModel):
    input_tensor = 'Cast:0'
    output_tensor = 'softmax:0'
    bottleneck_tensor = 'pool_3:0'
    image_tensor = 'DecodeJpeg/contents:0'

    def __init__(self, path=INCEPTIONV3_PB_PATH, name='FrozenInceptionV3', sess=None):
        path = ensure_inception_v3_pb(path)

        with TFSession(sess) as sess:
            super().__init__(path, name=name, sess=sess)
