import numpy as np
import os
import pandas as pd
import tensorflow as tf

from tfwrapper import logger
from tfwrapper import TFSession
from .pretrained_model import PretrainedModel

from .utils import INCEPTION_PB_PATH
from .utils import download_inceptionv3


class InceptionV3(PretrainedModel):
    DEFAULT_INPUT_LAYER = 'Cast:0'
    DEFAULT_FEATURES_LAYER = 'pool_3:0'

    core_layer_names = [
        {'name': 'DecodeJpeg/contents:0', 'input': [-1, -1, -1, 3], 'output': [-1, 299, 299, 3]},
        {'name': 'Cast:0', 'input': [-1, 299, 299, 3], 'output': [-1, 149, 149, 3]},
        {'name': 'conv:0', 'input': [-1, 149, 149, 3], 'output': [-1, 147, 147, 32]},
        {'name': 'conv_1:0', 'input': [-1, 149, 149, 32], 'output': [-1, 147, 147, 64]},
        {'name': 'conv_2:0', 'input': [-1, 147, 147, 64], 'output': [-1, 147, 147, 64]},
        {'name': 'pool:0', 'input': [-1, 147, 147, 64], 'output': [-1, 73, 73, 64]},
        {'name': 'conv_3:0', 'input': [-1, 73, 73, 64], 'output': [-1, 71, 71, 80]},
        {'name': 'conv_4:0', 'input': [-1, 71, 71, 80], 'output': [-1, 71, 71, 192]},
        {'name': 'pool_1:0', 'input': [-1, 71, 71, 192] ,'output': [-1, 35, 35, 192]},
        {'name': 'mixed/join:0', 'input': [-1, 35, 35, 192], 'output': [-1, 35, 35, 256]},
        {'name': 'mixed_1/join:0', 'input': [-1, 35, 35, 256], 'output': [-1, 35, 35, 288]},
        {'name': 'mixed_2/join:0', 'input': [-1, 35, 35, 288], 'output': [-1, 35, 35, 288]},
        {'name': 'mixed_3/join:0', 'input': [-1, 35, 35, 288], 'output': [-1, 17, 17, 768]},
        {'name': 'mixed_4/join:0', 'input': [-1, 17, 17, 768], 'output': [-1, 17, 17, 768]},
        {'name': 'mixed_5/join:0', 'input': [-1, 17, 17, 768], 'output': [-1, 17, 17, 768]},
        {'name': 'mixed_6/join:0', 'input': [-1, 17, 17, 768], 'output': [-1, 17, 17, 768]},
        {'name': 'mixed_7/join:0', 'input': [-1, 17, 17, 768], 'output': [-1, 17, 17, 768]},
        {'name': 'mixed_8/join:0', 'input': [-1, 17, 17, 768], 'output': [-1, 8, 8, 2048]},
        {'name': 'mixed_9/join:0', 'input': [-1, 8, 8, 2048], 'output': [-1, 8, 8, 2048]},
        {'name': 'mixed_10/join:0', 'input': [-1, 8, 8, 2048], 'output': [-1, 8, 8, 2048]},
        {'name': 'pool_3:0', 'input': [-1, 8, 8, 2048], 'output': [-1, 1, 1, 2048]},
        {'name': 'softmax:0', 'input': [-1, 1, 1, 2048], 'output': [-1, 1000]},
    ]

    def __init__(self,graph_file=INCEPTION_PB_PATH):
        graph_file = download_inceptionv3(graph_file, verbose=True)

        if not os.path.isfile(graph_file):
            raise Exception('Invalid path to inception v3 pb file')

        self.graph_file = graph_file

        with tf.gfile.FastGFile(graph_file,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def,name='')
            
            super().__init__(tf.get_default_graph())

    def run_op(self, to_layer, from_layer, data, sess=None):
        with TFSession(sess, self.graph) as sess:
            logger.info('Extracting features from layer ' + from_layer + ' to ' + to_layer)
            to_tensor = self.graph.get_tensor_by_name(to_layer)
            
            try:
                feature = sess.run(to_tensor,{from_layer: data})
                return feature[0]
            except Exception as e:
                logger.warning('Unable to extract feature')
                logger.warning(str(e))
                raise e

    def extract_features(self, img, layer=DEFAULT_FEATURES_LAYER, sess=None):
        with TFSession(sess, self.graph) as sess:
            return self.run_op(layer, self.DEFAULT_INPUT_LAYER, img, sess=sess)

    def extract_features_from_file(self, filename, layer=DEFAULT_FEATURES_LAYER, sess=None):
        with TFSession(sess, self.graph) as sess:
            try:
                image_data = tf.gfile.FastGFile(filename, 'rb').read()
                feature = self.run_op(layer, 'DecodeJpeg/contents:0', image_data, sess=sess)
                
                return feature
            except Exception as e:
                logger.warning('Unable to read image %' % filename)
                logger.warning(str(e))

                return None
