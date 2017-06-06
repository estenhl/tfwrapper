import os
import numpy as np
import tensorflow as tf

from abc import ABC

from tfwrapper import logger
from tfwrapper import TFSession


class FrozenModel(ABC):
    def __init__(self, path, *, input_tensor, output_tensor, bottleneck_tensor, name='FrozenModel', sess=None):
        if not os.path.isfile(path):
            raise Exception('Invalid path to inception v3 pb file')

        self.graph_file = path
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.bottleneck_tensor = bottleneck_tensor

        with TFSession(sess) as sess:
            with tf.gfile.FastGFile(path,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def,name='')
                
                self.graph = sess.graph

    def run_op(self, X, *, src, dest, sess=None):
        with TFSession(sess) as sess:
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

        with TFSession(sess) as sess:
            return self.run_op(src=self.input_tensor, dest=dest, X=X, sess=sess)

    def extract_bottleneck_features(self, X, sess=None):
        with TFSession(sess) as sess:
            features = self.extract_features(X, sess=sess)
            return np.squeeze(features)

    def predict(self, X, sess=None):
        with TFSession(sess) as sess:
            return self.run_op(src=self.input_tensor, dest=self.output_tensor, X=X, sess=sess)