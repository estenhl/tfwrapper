import numpy as np

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

    def extract_features(self, X, dest=None, sess=None):
        if dest is None:
            dest = self.bottleneck_tensor

        # Returns a single vector if X is a single image
        if len(X.shape) == 3:
            return self.run_op(X, src=self.input_tensor, dest=dest, sess=sess)

        # Returns a list of features if X is a list of images
        features = []
        for i in range(len(X)):
            features.append(self.run_op(X[i], src=self.input_tensor, dest=dest, sess=sess)[0])

        return np.asarray(features)