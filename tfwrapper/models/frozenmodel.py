import os
import numpy as np
import tensorflow as tf

from abc import ABC

from tfwrapper import logger
from tfwrapper import TFSession
from tfwrapper.models import PredictiveClassificationModel
from tfwrapper.models.utils import save_serving as save
from tfwrapper.utils.exceptions import InvalidArgumentException
from tfwrapper.utils.exceptions import raise_exception
from tfwrapper.utils.data import get_subclass_by_name

class FrozenModel(PredictiveClassificationModel):
    def __init__(self, path, name='FrozenModel', sess=None):
        if not os.path.isfile(path):
            raise_exception('Invalid path %s to pb file' % path, InvalidArgumentException)

        self.graph_file = path
        self.name = name

        with TFSession(sess) as sess:
            with tf.gfile.FastGFile(path,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def,name='')
                
                self.graph = sess.graph

    @classmethod
    def from_type(cls, classname, path=None, name='FrozenModel', sess=None):
        with TFSession(sess) as sess:
            subclass = get_subclass_by_name(cls, classname)
            return subclass(name=name, sess=sess)

    def run_op(self, X, *, src, dest, sess=None):
        with TFSession(sess, self.graph) as sess:
            logger.debug('Running operation from layer ' + src + ' to ' + dest)

            if type(dest) is str:
                to_tensor = self.graph.get_tensor_by_name(dest)

            try:
                return sess.run(dest, {src: X})
            except Exception as e:
                logger.warning('Unable to extract feature')
                logger.warning(str(e))
                raise e

    def extract_features(self, X, dest=None, sess=None):
        if dest is None:
            dest = self.bottleneck_tensor

        return self.run_op(X, src=self.input_tensor, dest=dest, sess=sess)

    def extract_bottleneck_features(self, X, sess=None):
        features = self.extract_features(X, dest=self.bottleneck_tensor, sess=sess)
        return np.squeeze(features)

    def predict(self, X, *, sess=None):
        return self.run_op(X, src=self.input_tensor, dest=self.output_tensor, sess=sess)

    def validate(self, X, y, *, sess=None):
        raise NotImplementedError('Validate is not implemented for frozen models')

    def save_serving(self, export_path, sess, over_write=False):
        in_tensor = sess.graph.get_tensor_by_name(self.input_tensor)
        out_tensor = sess.graph.get_tensor_by_name(self.bottleneck_tensor)

        save(export_path, in_tensor, out_tensor, sess, over_write=over_write)
